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

#pragma once

#include "dbstorage.h"

#include <atomic>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Database {

/// CompactDBKey is a compact version of database key, which requires less memory
/// by allocaing the minimal needed memory to store stone positions on the fly.
struct CompactDBKey
{
    Rule      rule;
    int8_t    boardWidth;
    int8_t    boardHeight;
    Color     sideToMove;
    uint16_t  numBlackStones;
    uint16_t  numWhiteStones;
    StonePos *stones;

    explicit CompactDBKey(const DBKey &key);
    explicit CompactDBKey(Rule      rule,
                          int       width,
                          int       height,
                          Color     sideToMove,
                          uint16_t  numBlackStones,
                          uint16_t  numWhiteStones,
                          StonePos *stones)
        : rule(rule)
        , boardWidth(width)
        , boardHeight(height)
        , sideToMove(sideToMove)
        , numBlackStones(numBlackStones)
        , numWhiteStones(numWhiteStones)
        , stones(stones)
    {}
    ~CompactDBKey();
    explicit operator DBKey() const;

    const StonePos *blackStonesBegin() const { return stones; }
    const StonePos *blackStonesEnd() const { return stones + numBlackStones; }
    const StonePos *whiteStonesBegin() const { return stones + numBlackStones; }
    const StonePos *whiteStonesEnd() const { return stones + numBlackStones + numWhiteStones; }

    /// Equal comparator of two database key.
    friend bool operator==(const CompactDBKey &a, const CompactDBKey &b)
    {
        return databaseKeyCompare(a, b) == 0;
    }
};

struct CompactDBKeyCmp
{
    using is_transparent = void;

    /// Less comparator of two database key.
    bool operator()(const CompactDBKey &a, const CompactDBKey &b) const
    {
        return databaseKeyCompare(a, b) < 0;
    }

    /// Heterogeneous comparasion between compact database key and normal database key.
    bool operator()(const CompactDBKey &a, const DBKey &b) const
    {
        return databaseKeyCompare(a, b) < 0;
    }
    bool operator()(const DBKey &a, const CompactDBKey &b) const
    {
        return databaseKeyCompare(a, b) < 0;
    }
};

/// YXDBStorage implements DBStorage interface for the Yixin-Database file format.
class YXDBStorage : public DBStorage
{
public:
    /// Creates a yixin database storage instance by opening the database file.
    /// If the file does not exist, it will creates the file.
    /// @param filePath The path of the database.
    /// @param compressedSave Enables LZ4-compressed file saving.
    /// @param saveOnClose Whether to save when YXDB is destroyed.
    /// @param backupOnSave When saving the file, copy previous file with the
    ///     same name to a new file with '_bak' postfix.
    /// @note Throws DBStorageError if failed to open the file.
    YXDBStorage(std::filesystem::path filePath,
                bool                  compressedSave,
                bool                  saveOnClose,
                int                   numBackupsOnSave      = 1,
                bool                  ignoreCorrupted       = false,
                bool                  journal               = false,
                int                   journalCompactPercent = 25);
    /// Close the yixin database. All unsaved records will be flushed to file.
    virtual ~YXDBStorage();

    /// Returns the current file path.
    std::filesystem::path getOpenedFilePath() const { return filePath; }

    // -------------------------------------------------------------------
    // Implements the DBStorage interface
    bool   get(const DBKey &key, DBRecord &record, DBRecordMask mask) noexcept override;
    void   set(const DBKey &key, const DBRecord &record, DBRecordMask mask) noexcept override;
    void   del(const DBKey &key) noexcept override;
    bool   flush() noexcept override;
    bool   compact() noexcept override;
    size_t size() noexcept override;
    Cursor scan(Cursor                                   cursor,
                size_t                                   count,
                std::vector<std::pair<DBKey, DBRecord>> &out) noexcept override;
    // -------------------------------------------------------------------

private:
    /// A write (set/del) that arrived while flush() was serializing the map.
    /// It is replayed onto the map once the file is written.
    struct PendingOp
    {
        DBKey        key;
        DBRecord     record;
        DBRecordMask mask;
        bool         isDelete;
    };

    /// One record of the sorted main table (20 bytes). The key's stones live
    /// in the key arena: one byte per stone ((x << 4) | y) on boards up to
    /// 16x16, otherwise the original two bytes (x, y). A narrow byte sorts
    /// exactly like the (x, y) pair memcmp() compares, so the main table keeps
    /// the order of databaseKeyCompare(). Text is rare (0.2% of records in the
    /// field) and sits in a side table keyed by keyOff.
    struct Entry
    {
        uint32_t     keyOff;
        uint16_t     numBlackStones;
        uint16_t     numWhiteStones;
        uint8_t      rule;
        int8_t       boardWidth;
        int8_t       boardHeight;
        uint8_t      flags;
        DBLabel      label;
        DBValue      value;
        DBDepthBound depthbound;
    };
    enum EntryFlag : uint8_t {
        ENTRY_WHITE_TO_MOVE = 1 << 0,
        ENTRY_DELETED       = 1 << 1,
        ENTRY_HAS_TEXT      = 1 << 2,
    };
    using OverlayMap = std::map<CompactDBKey, DBRecord, CompactDBKeyCmp>;

    /// Append-only storage of packed stones, in 1MB blocks: growing never
    /// moves existing keys, so loading has no reallocation peak.
    class KeyArena
    {
    public:
        static constexpr uint32_t BlockBits = 20;
        static constexpr uint32_t BlockSize = 1u << BlockBits;
        const uint8_t            *at(uint32_t off) const
        {
            return blocks[off >> BlockBits].get() + (off & (BlockSize - 1));
        }
        /// Reserves n contiguous bytes. Returns false if the arena is full (4GB).
        bool allocate(uint32_t n, uint32_t &off, uint8_t *&ptr);
        void clear()
        {
            blocks.clear();
            used = BlockSize;
        }
        size_t memoryBytes() const { return blocks.size() * size_t(BlockSize); }

    private:
        std::vector<std::unique_ptr<uint8_t[]>> blocks;
        uint32_t                                used = BlockSize;  // used bytes of the last block
    };

    std::filesystem::path filePath;
    // The records: a sorted vector loaded from the file, plus new keys added
    // since in a small map. Updates of keys in the main table happen in place,
    // deletes only flag them. mergeOverlay() folds the map back in.
    std::vector<Entry>                        entries;
    KeyArena                                  keyArena;
    std::unordered_map<uint32_t, std::string> entryTexts;  // keyOff -> utf-8 text
    size_t                                    numLiveEntries = 0;
    OverlayMap                                overlay;
    std::shared_mutex                         mutex;
    int                                               numBackupsOnSave;
    bool                                              compressedSave;
    bool                                              saveOnClose;
    bool                                              dirty;

    // flush() serializes under a *shared* lock so that search threads keep
    // reading the database while it is being saved. Writes arriving during
    // that window are queued here instead of waiting for the exclusive lock.
    // Lock order: pendingMutex -> mutex (writers and flush); readers never
    // hold mutex while taking pendingMutex.
    std::mutex             pendingMutex;
    std::mutex             flushMutex;  // one flush at a time
    bool                   saving = false;  // guarded by pendingMutex
    std::atomic<bool>      savingHint {false};  // lock-free fast path for get()
    std::vector<PendingOp> pendingOps;  // guarded by pendingMutex

    // Differential save (see the journal section of the .cpp).
    bool        journalEnabled;
    int         journalCompactPercent;
    bool        journalReplaying = false;
    bool        fullSaveRequired = false;  // next flush rewrites the file (flushMutex)
    std::string journalBuf;                // ops since the last flush (unique lock)
    uint32_t    journalOps       = 0;      // ops in journalBuf (unique lock)
    uint64_t    journalValidSize = 0;      // good bytes of <db>.journal, 0 = none (flushMutex)
    uint64_t    baseSize         = 0;      // identity of the file the journal
    uint64_t    baseTailHash     = 0;      // extends (flushMutex)

    /// Loads all data from current stream into memory.
    void load(std::istream &is, bool ignoreCorrupted);
    /// Save the current in-memory data to the opened file.
    void save(std::ostream &os) noexcept;
    /// Writes the whole map to `path`. Caller holds at least a shared lock.
    bool writeFile(const std::filesystem::path &path) noexcept;
    /// flush()/compact(). Caller holds nothing.
    bool flushImpl(bool compact) noexcept;
    /// Rewrites the whole file (atomic, non-blocking) and resets the journal.
    /// Caller holds flushMutex.
    bool fullSave() noexcept;

    // Journal. Caller holds flushMutex (or is the constructor).
    std::filesystem::path journalPath() const;
    static bool fileIdentity(const std::filesystem::path &path, uint64_t &size, uint64_t &hash);
    void        replayJournal();
    bool        appendJournal(const std::string &buf, uint32_t ops);
    bool        resetJournal();
    // Record one change into journalBuf. Caller holds the unique lock.
    void journalPut(const DBKey &key, const DBRecord &record);
    void journalDel(const DBKey &key);
    /// Apply one write to the map. Caller holds the unique lock.
    void applyLocked(const DBKey &key, const DBRecord &record, DBRecordMask mask, bool isDelete);

    // Main table helpers. Caller holds the lock (shared for reads).
    static bool isNarrow(int width, int height) { return width <= 16 && height <= 16; }
    /// Three-way compare of a main entry against any key with a `stones` array,
    /// in databaseKeyCompare() order.
    template <typename Key>
    int compareEntry(const Entry &e, const Key &key) const;
    /// Index of the first main entry not less than key (deleted ones included).
    template <typename Key>
    size_t lowerBound(const Key &key) const;
    template <typename Key>
    size_t upperBound(const Key &key) const;
    /// Index of the live main entry equal to key, or SIZE_MAX.
    size_t findEntry(const DBKey &key, bool includeDeleted = false) const;
    DBRecord entryRecord(const Entry &e) const;
    void     setEntryRecord(Entry &e, const DBRecord &record);
    DBKey    entryKey(const Entry &e) const;
    /// Packs key into a new main entry (record fields unset). False if it does
    /// not fit the narrow encoding (a coordinate outside 0..15) or the arena.
    template <typename Key>
    bool makeEntry(const Key &key, Entry &e);
    /// Folds the overlay into the sorted main table and drops deleted entries
    /// (their arena bytes stay until the next load). Caller holds the unique lock.
    void mergeOverlay();
    /// Writes one key / record body / record in the yixindb layout.
    template <typename Put, typename Key>
    static void putKey(Put &put, const Key &key);
    template <typename Put>
    static void putBody(Put &put, const DBRecord &record);
    template <typename Put, typename Key>
    static void putRecord(Put &put, const Key &key, const DBRecord &record)
    {
        putKey(put, key);
        putBody(put, record);
    }
};

}  // namespace Database
