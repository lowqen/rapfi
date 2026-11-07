/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "yxdbstorage.h"

#include "../core/compressor.h"
#include "../core/iohelper.h"
#include "../core/utils.h"
#include "dbconfig.h"
#include "dbtypes.h"

#include <cstring>
#include <fstream>
#include <mutex>
#include <string>

namespace Database {

CompactDBKey::CompactDBKey(const DBKey &key)
    : rule(key.rule)
    , boardWidth(key.boardWidth)
    , boardHeight(key.boardHeight)
    , sideToMove(key.sideToMove)
    , numBlackStones(key.numBlackStones)
    , numWhiteStones(key.numWhiteStones)
{
    stones = new StonePos[numBlackStones + numWhiteStones];
    std::copy(key.blackStonesBegin(), key.whiteStonesEnd(), stones);
}

CompactDBKey::~CompactDBKey()
{
    delete[] stones;
}

CompactDBKey::operator DBKey() const
{
    DBKey key;
    key.rule           = rule;
    key.boardWidth     = boardWidth;
    key.boardHeight    = boardHeight;
    key.sideToMove     = sideToMove;
    key.numBlackStones = numBlackStones;
    key.numWhiteStones = numWhiteStones;
    std::copy(stones, stones + numBlackStones + numWhiteStones, key.stones);
    return key;
}

YXDBStorage::YXDBStorage(std::filesystem::path filePath,
                         bool                  compressedSave,
                         bool                  saveOnClose,
                         int                   numBackupsOnSave,
                         bool                  ignoreCorrupted)
    : filePath(filePath)
    , numBackupsOnSave(numBackupsOnSave)
    , compressedSave(compressedSave)
    , saveOnClose(saveOnClose)
    , dirty(false)
{
    // A leftover from a save that was killed midway: the real file is intact.
    {
        std::error_code ec;
        std::filesystem::path tmpPath = filePath;
        tmpPath += ".tmp";
        std::filesystem::remove(tmpPath, ec);
    }

    if (!std::filesystem::exists(filePath))
        return;

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open() || !file)
        throw DBStorageError("Failed to open YXDB file at " + pathToConsoleString(filePath));

    MESSAGEL("DATABASE LOAD START " + pathToConsoleString(filePath));

    // Check LZ4 file magic to choose a compress type
    constexpr uint32_t LZ4FrameMagic = 0x184D2204;  // little-endian LZ4 frame magic number

    // A failed read (file shorter than 4 bytes) leaves the stream failed, which the
    // compressor stream check below turns into a DBStorageError.
    uint32_t magic = 0;
    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    file.seekg(0);

    {
        Compressor    compressor(static_cast<std::istream &>(file),
                              magic == LZ4FrameMagic ? Compressor::Type::LZ4_DEFAULT
                                                        : Compressor::Type::NO_COMPRESS);
        std::istream *istreamPtr = compressor.openInputStream();
        if (!istreamPtr || !*istreamPtr)
            throw DBStorageError("Failed to open LZ4 compressed YXDB file "
                                 + pathToConsoleString(filePath));

        load(*istreamPtr, ignoreCorrupted);
    }

    MESSAGEL("DATABASE LOAD DONE");
}

YXDBStorage::~YXDBStorage()
{
    if (saveOnClose)
        flush();
}

bool YXDBStorage::get(const DBKey &key, DBRecord &record, DBRecordMask mask) noexcept
{
    // Writes queued by a running flush() are newer than the map. Collect the
    // ones for this key before taking the map lock (lock order, see header).
    // Replaying an op the flush already applied is harmless: every op is
    // idempotent when the whole queued sequence is replayed in order.
    std::vector<PendingOp> ops;
    if (savingHint.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> pendingLock(pendingMutex);
        for (const PendingOp &op : pendingOps)
            if (databaseKeyCompare(op.key, key) == 0)
                ops.push_back(op);
    }

    std::shared_lock<std::shared_mutex> readerLock(mutex);

    auto it = recordsMap.find(key);
    if (ops.empty()) {
        if (it == recordsMap.end())
            return false;
        record.update(it->second, mask);
        return true;
    }

    bool     exists = it != recordsMap.end();
    DBRecord merged = exists ? it->second : DBRecord {};
    readerLock.unlock();
    for (const PendingOp &op : ops) {
        if (op.isDelete)
            exists = false;
        else if (exists)
            merged.update(op.record, op.mask);
        else
            merged = op.record, exists = true;
    }
    if (exists)
        record.update(merged, mask);
    return exists;
}

void YXDBStorage::applyLocked(const DBKey      &key,
                              const DBRecord   &record,
                              DBRecordMask      mask,
                              bool              isDelete)
{
    auto it = recordsMap.find(key);
    if (isDelete) {
        if (it != recordsMap.end()) {
            recordsMap.erase(it);
            dirty = true;
        }
        return;
    }

    if (it != recordsMap.end())
        it->second.update(record, mask);
    else
        recordsMap.insert(std::make_pair(key, record));
    dirty = true;
}

void YXDBStorage::set(const DBKey &key, const DBRecord &record, DBRecordMask mask) noexcept
{
    std::unique_lock<std::mutex> pendingLock(pendingMutex);
    if (saving) {
        pendingOps.push_back({key, record, mask, false});
        return;
    }
    // Take the map lock before releasing pendingMutex: flush() then cannot
    // start serializing between our check and our write.
    std::unique_lock<std::shared_mutex> writerLock(mutex);
    pendingLock.unlock();
    applyLocked(key, record, mask, false);
}

void YXDBStorage::del(const DBKey &key) noexcept
{
    std::unique_lock<std::mutex> pendingLock(pendingMutex);
    if (saving) {
        pendingOps.push_back({key, DBRecord {}, RECORD_MASK_NONE, true});
        return;
    }
    std::unique_lock<std::shared_mutex> writerLock(mutex);
    pendingLock.unlock();
    applyLocked(key, DBRecord {}, RECORD_MASK_NONE, true);
}

bool YXDBStorage::writeFile(const std::filesystem::path &path) noexcept
{
    std::ofstream file;
    std::vector<char> fileBuffer(1 << 20);
    file.rdbuf()->pubsetbuf(fileBuffer.data(), fileBuffer.size());
    file.open(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open())
        return false;

    {
        Compressor    compressor(static_cast<std::ostream &>(file),
                              compressedSave ? Compressor::Type::LZ4_DEFAULT
                                                : Compressor::Type::NO_COMPRESS);
        std::ostream *ostreamPtr = compressor.openOutputStream();
        if (!ostreamPtr || !*ostreamPtr)
            return false;
        save(*ostreamPtr);
        if (!*ostreamPtr)
            return false;
    }  // the compressor writes the LZ4 end mark here

    file.close();
    return !file.fail();
}

bool YXDBStorage::flush() noexcept
{
    std::lock_guard<std::mutex> flushLock(flushMutex);

    {
        std::lock_guard<std::mutex>         pendingLock(pendingMutex);
        std::shared_lock<std::shared_mutex> readerLock(mutex);  // waits for in-flight writers
        if (!dirty && std::filesystem::exists(filePath))
            return true;
        dirty  = false;
        saving = true;
        savingHint.store(true, std::memory_order_release);
    }

    MESSAGEL("DATABASE SAVE START " + pathToConsoleString(filePath));

    // 1. Write the complete new file next to the old one. Search threads keep
    //    reading (shared lock); their writes are queued in pendingOps.
    std::filesystem::path tmpPath = filePath;
    tmpPath += ".tmp";
    bool ok;
    {
        std::shared_lock<std::shared_mutex> readerLock(mutex);
        ok = writeFile(tmpPath);
    }

    // 2. Only a fully written file replaces the old one. The old file stays
    //    untouched if anything above failed (disk full, killed mid-write).
    std::error_code ec;
    if (ok) {
        auto makeBackupFilePath = [this](int index) -> std::filesystem::path {
            std::filesystem::path backupPath = filePath;
            std::string           indexStr   = std::to_string(index);
            auto                  fn_noext   = backupPath.stem().concat("_bak").concat(indexStr);
            auto                  ext        = backupPath.extension();
            backupPath.replace_filename(fn_noext);
            backupPath.replace_extension(ext);
            return backupPath;
        };

        if (numBackupsOnSave && std::filesystem::exists(filePath)) {
            for (int i = numBackupsOnSave; i > 1; i--) {
                std::filesystem::remove(makeBackupFilePath(i), ec);
                std::filesystem::rename(makeBackupFilePath(i - 1), makeBackupFilePath(i), ec);
            }
            // Keep the current file in place while making backup #1, so the
            // database path never points at nothing: a hard link costs nothing;
            // where links are refused (Android apps, FAT) copy it; only if that
            // fails too, fall back to the original move.
            auto bak1 = makeBackupFilePath(1);
            std::filesystem::remove(bak1, ec);
            std::filesystem::create_hard_link(filePath, bak1, ec);
            if (ec)
                std::filesystem::copy_file(filePath, bak1, ec);
            if (ec) {
                std::filesystem::remove(bak1, ec);
                std::filesystem::rename(filePath, bak1, ec);
            }
        }

        std::filesystem::rename(tmpPath, filePath, ec);  // replaces atomically on POSIX
        if (ec) {
            // Some Windows runtimes refuse to rename over an existing file.
            std::error_code ec2;
            std::filesystem::remove(filePath, ec2);
            std::filesystem::rename(tmpPath, filePath, ec);
        }
        ok = !ec;
    }
    if (!ok)
        std::filesystem::remove(tmpPath, ec);

    // 3. Replay the writes that arrived meanwhile; they are not in the file.
    {
        std::lock_guard<std::mutex>         pendingLock(pendingMutex);
        std::unique_lock<std::shared_mutex> writerLock(mutex);
        for (const PendingOp &op : pendingOps)
            applyLocked(op.key, op.record, op.mask, op.isDelete);
        pendingOps.clear();
        pendingOps.shrink_to_fit();
        if (!ok)
            dirty = true;
        saving = false;
        savingHint.store(false, std::memory_order_release);
    }

    if (ok) {
        MESSAGEL("DATABASE SAVE DONE");
        return true;
    }
    ERRORL("Failed to save YXDB file at " + pathToConsoleString(filePath));
    return false;
}

size_t YXDBStorage::size() noexcept
{
    std::shared_lock<std::shared_mutex> readerLock(mutex);
    return recordsMap.size();
}

YXDBStorage::Cursor YXDBStorage::scan(Cursor                                   cursor,
                                      size_t                                   count,
                                      std::vector<std::pair<DBKey, DBRecord>> &out) noexcept
{
    std::shared_lock<std::shared_mutex> readerLock(mutex);

    // Resume right after the last key of the previous batch (O(log n), and
    // insertions/deletions between batches can not shift the scan position)
    auto it = cursor ? recordsMap.upper_bound(cursor.lastKey) : recordsMap.begin();

    bool  readAny = false;
    DBKey lastKey;
    while (count > 0 && it != recordsMap.end()) {
        lastKey = DBKey(it->first);
        out.emplace_back(lastKey, it->second);
        it++;
        count--;
        readAny = true;
    }

    if (it == recordsMap.end())
        return Cursor {};
    return readAny ? Cursor {lastKey} : cursor;  // count == 0 call: position unchanged
}

void YXDBStorage::load(std::istream &is, bool ignoreCorrupted)
{
    uint32_t numRecords;
    is.read(reinterpret_cast<char *>(&numRecords), sizeof(numRecords));

    std::vector<int8_t> byteBuffer;
    byteBuffer.resize(1024);  // reserve initial space

    // Read an int16 with memcpy, as value/depthbound sit at odd buffer offsets
    auto readInt16At = [&byteBuffer](size_t offset) {
        int16_t v;
        std::memcpy(&v, &byteBuffer[offset], sizeof(v));
        return v;
    };

    bool              isUTF8       = false;  // Is this database UTF-8 encoded?
    const std::string utf8Metadata = "charset=\"UTF-8\"";

    auto hint = recordsMap.begin();
    for (uint32_t recordIdx = 0; recordIdx < numRecords; recordIdx++) {
        // Read record key
        uint16_t numKeyBytes;
        is.read(reinterpret_cast<char *>(&numKeyBytes), sizeof(numKeyBytes));
        if (numKeyBytes == 0)
            continue;
        if (numKeyBytes > byteBuffer.size())
            byteBuffer.resize(numKeyBytes);
        is.read(reinterpret_cast<char *>(byteBuffer.data()), numKeyBytes);

        // Parse record key
        Rule rule = static_cast<Rule>(byteBuffer[0]);
        if (rule >= RULE_NB) {
            if (ignoreCorrupted)
                continue;
            throw DBStorageCorruptedRecordError(pathToConsoleString(filePath),
                                                "with invalid rule at index "
                                                    + std::to_string(recordIdx));
        }

        int boardXLen = byteBuffer[1], boardYLen = byteBuffer[2];
        if ((unsigned)boardXLen > MAX_BOARD_SIZE || (unsigned)boardYLen > MAX_BOARD_SIZE) {
            if (ignoreCorrupted)
                continue;
            throw DBStorageCorruptedRecordError(pathToConsoleString(filePath),
                                                "with invalid board size at index "
                                                    + std::to_string(recordIdx));
        }

        uint16_t numStones = (numKeyBytes - 3) / 2;
        if (numStones > boardXLen * boardYLen) {
            if (ignoreCorrupted)
                continue;
            throw DBStorageCorruptedRecordError(pathToConsoleString(filePath),
                                                "with invalid number of stones at index "
                                                    + std::to_string(recordIdx));
        }

        uint16_t numBlackStones = (numStones + 1) / 2;
        uint16_t numWhiteStones = numStones / 2;
        Color    sideToMove     = numBlackStones == numWhiteStones ? BLACK : WHITE;

        auto   stones   = std::make_unique<StonePos[]>(numStones);
        size_t stoneIdx = 0;
        for (uint16_t blackIdx = 0; blackIdx < numBlackStones; blackIdx++) {
            int8_t x = byteBuffer[3 + blackIdx * 2];
            int8_t y = byteBuffer[4 + blackIdx * 2];
            if (x == -1 && y == -1)  // the last pass move
                break;
            else if (x >= 0 && y >= 0 && x < boardXLen && y < boardYLen)
                stones[stoneIdx++] = {x, y};
            else if (ignoreCorrupted)
                goto next_record;
            else
                throw DBStorageCorruptedRecordError(pathToConsoleString(filePath),
                                                    "with invalid black pos at index "
                                                        + std::to_string(recordIdx));
        }
        for (uint16_t whiteIdx = 0; whiteIdx < numWhiteStones; whiteIdx++) {
            int8_t x = byteBuffer[3 + (numBlackStones + whiteIdx) * 2];
            int8_t y = byteBuffer[4 + (numBlackStones + whiteIdx) * 2];
            if (x == -1 && y == -1)  // the last pass move
                break;
            else if (x >= 0 && y >= 0 && x < boardXLen && y < boardYLen)
                stones[stoneIdx++] = {x, y};
            else if (ignoreCorrupted)
                goto next_record;
            else
                throw DBStorageCorruptedRecordError(pathToConsoleString(filePath),
                                                    "with invalid white pos at index "
                                                        + std::to_string(recordIdx));
        }

        // Read record message
        uint16_t numRecordBytes;
        is.read(reinterpret_cast<char *>(&numRecordBytes), sizeof(numRecordBytes));
        if (numRecordBytes > byteBuffer.size())
            byteBuffer.resize(numRecordBytes);
        is.read(reinterpret_cast<char *>(byteBuffer.data()), numRecordBytes);

        // Parse metadata record
        if (boardXLen == 0 && boardYLen == 0 && numRecordBytes >= 5 + utf8Metadata.length()) {
            std::string metadata {reinterpret_cast<char *>(&byteBuffer[5]),
                                  static_cast<size_t>(numRecordBytes - 5)};
            if (metadata.rfind(utf8Metadata, 0) == 0)
                isUTF8 = true;
            continue;
        }

        // Emplace db key and db record in map
        hint = recordsMap.emplace_hint(
            hint,
            std::piecewise_construct,
            std::forward_as_tuple(rule,
                                  boardXLen,
                                  boardYLen,
                                  sideToMove,
                                  numBlackStones,
                                  numWhiteStones,
                                  stones.release()),
            std::forward_as_tuple(DBRecord {
                numRecordBytes > 0 ? static_cast<DBLabel>(byteBuffer[0]) : DBLabel(0),
                numRecordBytes > 2 ? DBValue(readInt16At(1)) : DBValue(0),
                numRecordBytes > 4 ? DBDepthBound(readInt16At(3)) : DBDepthBound(0),
                numRecordBytes > 5
                    ? (isUTF8 ? std::string {reinterpret_cast<char *>(&byteBuffer[5]),
                                             static_cast<size_t>(numRecordBytes - 5)}
                              : LegacyFileCPToUTF8(
                                    std::string {reinterpret_cast<char *>(&byteBuffer[5]),
                                                 static_cast<size_t>(numRecordBytes - 5)},
                                    DatabaseCfg.legacyFileCodePage))
                    : std::string {}}));

    next_record:;
    }

    // Force convert old format to new utf-8 format
    if (!isUTF8)
        dirty = true;
}

void YXDBStorage::save(std::ostream &os) noexcept
{
    uint32_t numRecords = recordsMap.size() + 1;
    os.write(reinterpret_cast<char *>(&numRecords), sizeof(numRecords));

    // Write a metadata record with board size equals to 0
    {
        // Record charset of this database file
        std::string metadata = "charset=\"UTF-8\"";

        uint16_t numMetaKeyBytes = 3;
        char     metaKeyData[3]  = {0};
        os.write(reinterpret_cast<char *>(&numMetaKeyBytes), sizeof(numMetaKeyBytes));
        os.write(reinterpret_cast<const char *>(metaKeyData), 3);

        uint16_t numRecordBytes    = 5 + metadata.length();
        char     metaRecordData[5] = {0};
        os.write(reinterpret_cast<char *>(&numRecordBytes), sizeof(numRecordBytes));
        os.write(reinterpret_cast<const char *>(metaRecordData),
                 sizeof(DBLabel) + sizeof(DBValue) + sizeof(DBDepthBound));
        os.write(metadata.c_str(), metadata.length());
    }

    // Records are tiny (~80 bytes); serialize them into a chunk and hand the
    // compressor large writes instead of ~6 small writes per record.
    constexpr size_t ChunkSize = 1 << 20;
    std::string      chunk;
    chunk.reserve(ChunkSize + 4096);
    auto put = [&chunk](const void *p, size_t n) {
        chunk.append(reinterpret_cast<const char *>(p), n);
    };

    const int8_t PassMove[2] = {-1, -1};
    for (auto it = recordsMap.cbegin(); it != recordsMap.cend(); it++) {
        const CompactDBKey &key    = it->first;
        const DBRecord     &record = it->second;

        // Serialize record key
        Color    normalSTM   = (key.numBlackStones + key.numWhiteStones) % 2 == 0 ? BLACK : WHITE;
        bool     addPassMove = normalSTM != key.sideToMove;
        uint16_t numKeyBytes =
            3 + 2 * (key.whiteStonesEnd() - key.blackStonesBegin() + addPassMove);

        put(&numKeyBytes, sizeof(numKeyBytes));
        put(&key, 3);
        put(key.blackStonesBegin(), (key.blackStonesEnd() - key.blackStonesBegin()) * 2);
        if (addPassMove && normalSTM == BLACK)
            put(PassMove, 2);
        put(key.whiteStonesBegin(), (key.whiteStonesEnd() - key.whiteStonesBegin()) * 2);
        if (addPassMove && normalSTM == WHITE)
            put(PassMove, 2);

        // Write record message
        if (record.isNull()) {
            uint16_t numRecordBytes = 0;
            put(&numRecordBytes, sizeof(numRecordBytes));
        }
        else {
            const std::string &utf8Text       = record.text;
            uint16_t           numRecordBytes = 5 + utf8Text.length();
            put(&numRecordBytes, sizeof(numRecordBytes));
            put(&record.label, sizeof(DBLabel));
            put(&record.value, sizeof(DBValue));
            put(&record.depthbound, sizeof(DBDepthBound));
            put(utf8Text.data(), utf8Text.length());
        }

        if (chunk.size() >= ChunkSize) {
            os.write(chunk.data(), chunk.size());
            chunk.clear();
        }
    }
    os.write(chunk.data(), chunk.size());
}

}  // namespace Database
