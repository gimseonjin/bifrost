package warp

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/framework/configstore"
	"github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/maximhq/bifrost/framework/vectorstore"
)

// Warp keeps its log index in one of the vector stores config.json configures, chosen by
// name in its settings (TableWarpConfig.VectorStore; "" is the primary vector_store). The
// choice is part of the configuration row every part of Warp reads before it touches a
// vector, so each read follows it: the service holds one vectorstore.Switchable that the
// searcher and topics use, and a read that finds a different choice than the one held swaps
// the store under them. Indexing, log deletes, retention and the storage limit take the store
// from the row they read, so each pairs a store with the namespace that same row names. A
// change saved on one node reaches the others with their next read. A choice that names no
// configured store, or one that could not be reached at startup, leaves Warp with no store,
// never another one.
//
// Moving the index to another store starts a new index there, which a backfill fills, and
// leaves the old one to be deleted. The save records it (TableWarpConfig.VectorStoreCleanup),
// and upkeep deletes Warp's vectors from it once vectorStoreCleanupGrace has passed, so a
// write already in flight on any node has landed, retrying on each pass while that store
// cannot be reached. Only the index in use is ever read, deleted from or swept.

// VectorStoreResolver finds the vector stores config.json configures; *vectorstore.Registry is one.
type VectorStoreResolver interface {
	Get(name string) (*vectorstore.NamedStore, bool)
	Primary() *vectorstore.NamedStore
	All() []*vectorstore.NamedStore
}

// WithVectorStores lets Warp keep its log index in any configured store, chosen in its
// settings. Without it, Warp uses the store WithVectorStore gives it and offers no choice.
func WithVectorStores(stores VectorStoreResolver) Option {
	return func(s *Service) { s.vectorStores = stores }
}

// VectorStoreUnavailableError is ErrNoVectorStore for a chosen store Warp cannot use.
type VectorStoreUnavailableError struct {
	Name   string
	Reason string // "is not configured in config.json", "could not be reached at startup", ...
}

func (e *VectorStoreUnavailableError) Error() string {
	return fmt.Sprintf("Warp's vector store %q %s", e.Name, e.Reason)
}

func (e *VectorStoreUnavailableError) Unwrap() error { return ErrNoVectorStore }

// ErrVectorStoreChoiceChanged refuses a save that read Warp's settings before another save
// changed its vector store: what it checked, and what it would record as left behind, were
// for a store the index is no longer in.
var ErrVectorStoreChoiceChanged = errors.New("warp: Warp's vector store was changed by another save; reload the settings and save again")

// ConnectedVectorStore is the store Warp's log index is kept in, as the settings page shows it.
type ConnectedVectorStore struct {
	Name    string `json:"name"`
	Type    string `json:"type"`
	Primary bool   `json:"primary"`
}

// vectorSelection is a choice of store: the name as stored ("" is the primary), the index
// built in it (TableWarpConfig.VectorIndexID), and what the name resolves to, nil when it
// names no configured store.
type vectorSelection struct {
	name    string
	indexID string
	entry   *vectorstore.NamedStore
}

// store is the open connection the selection reaches, or nil.
func (v *vectorSelection) store() vectorstore.VectorStore {
	if v == nil {
		return nil
	}
	return connectedStore(v.entry)
}

// connectedStore is an entry's open connection, or nil.
func connectedStore(entry *vectorstore.NamedStore) vectorstore.VectorStore {
	if entry == nil || entry.Status != vectorstore.StoreConnected {
		return nil
	}
	return entry.Store
}

// resolveEntry finds the configured store a stored name reaches, nil when none.
func (s *Service) resolveEntry(name string) *vectorstore.NamedStore {
	if s.vectorStores == nil {
		return nil
	}
	if name == "" {
		return s.vectorStores.Primary()
	}
	entry, _ := s.vectorStores.Get(name)
	return entry
}

// resolveSelection finds what a stored name reaches.
func (s *Service) resolveSelection(name string) *vectorSelection {
	return &vectorSelection{name: name, entry: s.resolveEntry(name)}
}

// rowSelection is the choice a configuration row makes.
func (s *Service) rowSelection(row *tables.TableWarpConfig) *vectorSelection {
	if row == nil {
		return s.resolveSelection("")
	}
	selection := s.resolveSelection(row.VectorStore)
	selection.indexID = row.VectorIndexID
	return selection
}

// storeForRow is the store a configuration row keeps the index in, or nil when it cannot be
// used now. Without a choice of stores it is the one store Warp was given.
func (s *Service) storeForRow(row *tables.TableWarpConfig) vectorstore.VectorStore {
	if s.vectorSlot == nil {
		return s.vectorStore
	}
	name := ""
	if row != nil {
		name = row.VectorStore
	}
	if held := s.selection.Load(); held != nil && held.name == name {
		return held.store()
	}
	return connectedStore(s.resolveEntry(name))
}

// followVectorStore makes Warp use the store a configuration row chooses, if it is not the
// one in use. It runs on every read of the row: the check is one atomic load and two string
// comparisons.
func (s *Service) followVectorStore(row *tables.TableWarpConfig) {
	if s.vectorSlot == nil {
		return
	}
	name, indexID := "", ""
	if row != nil {
		name, indexID = row.VectorStore, row.VectorIndexID
	}
	if held := s.selection.Load(); held != nil && held.name == name && held.indexID == indexID {
		return
	}
	// The store and the choice that names it change together: two reads choosing different
	// stores must not leave the choice naming one while calls reach the other, since the
	// check above would then never correct it. A read of an older row can still move the
	// choice back briefly; the next read of the row moves it again.
	s.selectionMu.Lock()
	defer s.selectionMu.Unlock()
	held := s.selection.Load()
	if held != nil && held.name == name && held.indexID == indexID {
		return
	}
	next := s.resolveSelection(name)
	next.indexID = indexID
	s.vectorSlot.Swap(next.store())
	s.selection.Store(next)
	if held != nil && held.name == name {
		return // the same store, with a new index in it
	}
	if next.store() == nil {
		s.warnf("%v; Warp is unavailable until it is fixed", unavailableVectorStore(next))
		return
	}
	s.infof("Warp's log index is now kept in vector store %q", next.entry.Name)
}

// currentVectorStore is the store Warp reads and writes now, or nil.
func (s *Service) currentVectorStore() vectorstore.VectorStore {
	if s.vectorSlot != nil {
		return s.vectorSlot.Current()
	}
	return s.vectorStore
}

// hasVectorStore reports whether Warp has a store to keep its index in now.
func (s *Service) hasVectorStore() bool {
	return s.currentVectorStore() != nil
}

// noVectorStore is the error for having no store: which chosen store is unavailable and why,
// when there is one to name.
func (s *Service) noVectorStore() error {
	if s.vectorSlot == nil {
		return ErrNoVectorStore
	}
	return unavailableVectorStore(s.selection.Load())
}

func unavailableVectorStore(selection *vectorSelection) error {
	if selection == nil {
		return ErrNoVectorStore
	}
	if selection.entry == nil {
		if selection.name == "" {
			return ErrNoVectorStore // no vector_store in config.json
		}
		return &VectorStoreUnavailableError{Name: selection.name, Reason: "is not configured in config.json"}
	}
	switch selection.entry.Status {
	case vectorstore.StoreFailed:
		return &VectorStoreUnavailableError{Name: selection.entry.Name, Reason: "could not be reached at startup"}
	case vectorstore.StoreDisabled:
		return &VectorStoreUnavailableError{Name: selection.entry.Name, Reason: "is disabled in config.json"}
	}
	return ErrNoVectorStore
}

// vectorStoreStatus is what the settings view says about a selection: whether Warp has a
// store, the store when it does, and why it has none when there is more to say than that.
func (s *Service) vectorStoreStatus(selection *vectorSelection) (connected bool, store *ConnectedVectorStore, reason string) {
	if s.vectorSlot == nil {
		return s.vectorStore != nil, nil, ""
	}
	if selection.store() == nil {
		if err := unavailableVectorStore(selection); err != ErrNoVectorStore {
			reason = err.Error()
		}
		return false, nil, reason
	}
	return true, &ConnectedVectorStore{Name: selection.entry.Name, Type: string(selection.entry.Type), Primary: selection.entry.Primary}, ""
}

// currentVectorIndex is the index Warp builds now, for backfill checkpoints; "" with no
// choice of stores, and for an index that never moved.
func (s *Service) currentVectorIndex() string {
	if s.vectorSlot == nil {
		return ""
	}
	if selection := s.selection.Load(); selection != nil {
		return selection.indexID
	}
	return ""
}

// checkpointIsForThisIndex reports whether a backfill checkpoint was written for the index
// Warp builds now. Each move to another store starts a new index, so a checkpoint from
// before one covers logs the index no longer has, even one written into the same store
// before the index left it and came back. A checkpoint from before stores could be chosen
// names no index, like one for an index that never moved.
func (s *Service) checkpointIsForThisIndex(indexID string) bool {
	return indexID == s.currentVectorIndex()
}

// newVectorIndexID names a new index.
func newVectorIndexID() string {
	var random [8]byte
	_, _ = rand.Read(random[:])
	return hex.EncodeToString(random[:])
}

// vectorStoreCleanupGrace is how long an index Warp moved away from is kept before upkeep
// deletes it: longer than any write already in flight when the move was saved (an index
// item is bounded by warpIndexItemTimeout), on this node or on one that has not read the
// move yet.
const vectorStoreCleanupGrace = 10 * time.Minute

// vectorStoreCleanup is an index Warp moved away from, for upkeep to delete.
type vectorStoreCleanup struct {
	Store     string    `json:"store"`    // the store's name when the index left it, for messages
	StoreID   string    `json:"store_id"` // where it keeps its data (NamedStore.ID), which is how it is found again
	InMemory  bool      `json:"in_memory,omitempty"`
	Namespace string    `json:"namespace"`
	Dimension int       `json:"dimension"`
	Since     time.Time `json:"since"` // when the index left it
}

func (c vectorStoreCleanup) same(other vectorStoreCleanup) bool {
	return c.StoreID == other.StoreID && c.Namespace == other.Namespace && c.Since.Equal(other.Since)
}

// pendingCleanups decodes the indexes a row leaves to upkeep; a value that does not decode is none.
func pendingCleanups(row *tables.TableWarpConfig) []vectorStoreCleanup {
	if row == nil || row.VectorStoreCleanup == nil || *row.VectorStoreCleanup == "" {
		return nil
	}
	var pending []vectorStoreCleanup
	if err := sonic.UnmarshalString(*row.VectorStoreCleanup, &pending); err != nil {
		return nil
	}
	return pending
}

func encodeCleanups(pending []vectorStoreCleanup) (*string, error) {
	if len(pending) == 0 {
		return nil, nil
	}
	encoded, err := sonic.MarshalString(pending)
	if err != nil {
		return nil, err
	}
	return &encoded, nil
}

// storedStoreID is where a row's chosen store keeps its data: as recorded when it was chosen,
// or, for a row from before that was recorded, what the name reaches now.
func (s *Service) storedStoreID(row *tables.TableWarpConfig) string {
	if row == nil {
		return ""
	}
	if row.VectorStoreID != "" {
		return row.VectorStoreID
	}
	if entry := s.resolveEntry(row.VectorStore); entry != nil {
		return entry.ID
	}
	return ""
}

// indexMoves reports whether choosing target moves the index out of the store a row keeps it
// in: a different name for a store with different data. A store renamed in config.json, or
// another name for a shared connection, is the same data and is not a move. Without a row
// nothing was indexed, so nothing moves.
func (s *Service) indexMoves(row *tables.TableWarpConfig, target *vectorSelection) bool {
	if s.vectorSlot == nil || row == nil || target.name == row.VectorStore {
		return false
	}
	from := s.storedStoreID(row)
	return from == "" || target.entry == nil || target.entry.ID != from
}

// leftBehind is the index a row keeps in its store, as a move leaves it for upkeep; ok is
// false when the row never indexed anything, or names a store that cannot be found again.
func (s *Service) leftBehind(row *tables.TableWarpConfig, at time.Time) (vectorStoreCleanup, bool) {
	if row == nil || row.EmbeddingModel == "" || row.EmbeddingDimension <= 0 {
		return vectorStoreCleanup{}, false
	}
	left := vectorStoreCleanup{Store: row.VectorStore, StoreID: s.storedStoreID(row), Namespace: normalizedNamespace(row.LogVectorStoreNamespace), Dimension: row.EmbeddingDimension, Since: at.UTC()}
	if entry := s.resolveEntry(row.VectorStore); entry != nil {
		left.Store, left.InMemory = entry.Name, entry.InMemory
	}
	return left, left.StoreID != ""
}

// storeByID finds a configured store by where it keeps its data, preferring one that is
// connected, so a store renamed since the index left it is still found.
func (s *Service) storeByID(id string) *vectorstore.NamedStore {
	if s.vectorStores == nil || id == "" {
		return nil
	}
	var found *vectorstore.NamedStore
	for _, entry := range s.vectorStores.All() {
		if entry == nil || entry.ID != id {
			continue
		}
		if connectedStore(entry) != nil {
			return entry
		}
		if found == nil {
			found = entry
		}
	}
	return found
}

// warpLogFilter matches Warp's own vectors.
var warpLogFilter = []vectorstore.Query{{Field: "warp_log", Operator: vectorstore.QueryOperatorEqual, Value: true}}

// deleteWarpIndex deletes Warp's vectors under namespace from a store. A namespace holding
// only Warp's vectors, as one Warp provisioned does, goes whole; one something else writes
// to as well keeps what is not Warp's. A namespace that is not there is already gone.
func deleteWarpIndex(ctx context.Context, store vectorstore.VectorStore, namespace string, dimension int) error {
	existing, err := store.ListNamespaces(ctx, namespace)
	if err != nil {
		return err
	}
	if !slices.Contains(existing, namespace) {
		return nil
	}
	// Opened first: an embedded Chromem store cannot count or filter a namespace this
	// process has not opened, as after a restart.
	if _, err := ensureWarpNamespace(ctx, store, namespace, dimension); err != nil {
		return err
	}
	all, err := store.CountVectors(ctx, namespace, nil)
	if err != nil {
		return err
	}
	warp, err := store.CountVectors(ctx, namespace, warpLogFilter)
	if err != nil {
		return err
	}
	if all == warp {
		return store.DeleteNamespace(ctx, namespace)
	}
	_, err = store.DeleteAll(ctx, namespace, warpLogFilter)
	return err
}

// deleteLeftIndexes deletes the indexes Warp moved away from once vectorStoreCleanupGrace has
// passed, and reports how many went. One whose store cannot be reached now waits for a later
// pass; one whose store is no longer configured waits for it to be configured again, except a
// memory-only store's, which went with the process that held it.
func (s *Service) deleteLeftIndexes(ctx context.Context, now time.Time) (int, error) {
	if s.store == nil || s.vectorSlot == nil {
		return 0, nil
	}
	row, err := s.store.GetWarpConfig(ctx)
	if err != nil || row == nil {
		return 0, err
	}
	pending := pendingCleanups(row)
	if len(pending) == 0 {
		return 0, nil
	}
	inUse, inUseNamespace := s.storedStoreID(row), normalizedNamespace(row.LogVectorStoreNamespace)
	var done []vectorStoreCleanup
	var failures []error
	for _, left := range pending {
		if now.Sub(left.Since) < vectorStoreCleanupGrace {
			continue
		}
		if left.StoreID == inUse && left.Namespace == inUseNamespace {
			done = append(done, left) // the index in use again, which the save that went back cleared
			continue
		}
		entry := s.storeByID(left.StoreID)
		if entry == nil {
			if left.InMemory {
				done = append(done, left)
			}
			continue
		}
		store := connectedStore(entry)
		if store == nil {
			continue
		}
		if err := deleteWarpIndex(ctx, store, left.Namespace, left.Dimension); err != nil {
			failures = append(failures, fmt.Errorf("delete Warp's earlier log index from vector store %q: %w", left.Store, err))
			continue
		}
		done = append(done, left)
	}
	if len(done) > 0 {
		if err := s.forgetCleanups(ctx, done); err != nil {
			failures = append(failures, fmt.Errorf("record deleted Warp log indexes: %w", err))
		}
	}
	return len(done), errors.Join(failures...)
}

// errNothingToWrite ends a merge that finds nothing to change.
var errNothingToWrite = errors.New("nothing to write")

// forgetCleanups takes deleted indexes off the row's list, from the row as stored at write time.
func (s *Service) forgetCleanups(ctx context.Context, done []vectorStoreCleanup) error {
	row := &tables.TableWarpConfig{}
	err := s.upsertWarpRow(ctx, row, func(stored *tables.TableWarpConfig) error {
		if stored == nil {
			return errNothingToWrite
		}
		*row = *stored
		remaining := slices.DeleteFunc(pendingCleanups(stored), func(left vectorStoreCleanup) bool {
			return slices.ContainsFunc(done, left.same)
		})
		encoded, err := encodeCleanups(remaining)
		if err != nil {
			return err
		}
		row.VectorStoreCleanup = encoded
		return nil
	})
	if errors.Is(err, errNothingToWrite) {
		return nil
	}
	return err
}

// warpConfigMerger is a config store that writes the Warp row with a merge from the row as stored
// at write time, under a lock (configstore.RDBConfigStore.UpsertWarpConfigMerging).
type warpConfigMerger interface {
	UpsertWarpConfigMerging(ctx context.Context, config *tables.TableWarpConfig, merge func(stored *tables.TableWarpConfig) error) error
}

// upsertWarpRow writes row after merge has completed it from the row as stored at write time,
// under the store's lock where it has one. A store without one reads the row again right
// before the write, which narrows the window.
func (s *Service) upsertWarpRow(ctx context.Context, row *tables.TableWarpConfig, merge func(stored *tables.TableWarpConfig) error) error {
	store := s.store
	if following, ok := store.(*selectionFollowingStore); ok {
		store = following.WarpStore
	}
	if merger, ok := store.(warpConfigMerger); ok {
		return merger.UpsertWarpConfigMerging(ctx, row, merge)
	}
	stored, err := store.GetWarpConfig(ctx)
	if err != nil {
		return err
	}
	if err := merge(stored); err != nil {
		return err
	}
	return store.UpsertWarpConfig(ctx, row)
}

// vectorStoreMove is what a save does to where the index is kept, settled before the write.
type vectorStoreMove struct {
	moves   bool
	indexID string // the new index's, when it moves
	at      time.Time
}

// mergeSave completes a save's row from the row as stored at write time. The save checked its
// choice of store against previous, its earlier read; a row whose choice changed since is
// refused, since what was checked and what would be left behind were for another store.
// Otherwise the index a move leaves behind joins the indexes upkeep deletes, taken from the
// row the write replaces, and the index moved to leaves them.
func (s *Service) mergeSave(row, previous *tables.TableWarpConfig, move vectorStoreMove) func(stored *tables.TableWarpConfig) error {
	choice := func(r *tables.TableWarpConfig) [2]string {
		if r == nil {
			return [2]string{}
		}
		return [2]string{r.VectorStore, r.VectorIndexID}
	}
	return func(stored *tables.TableWarpConfig) error {
		if choice(stored) != choice(previous) {
			return ErrVectorStoreChoiceChanged
		}
		pending := pendingCleanups(stored)
		row.VectorIndexID = ""
		if stored != nil {
			row.VectorIndexID = stored.VectorIndexID
		}
		if move.moves {
			row.VectorIndexID = move.indexID
			namespace := normalizedNamespace(row.LogVectorStoreNamespace)
			pending = slices.DeleteFunc(pending, func(left vectorStoreCleanup) bool {
				return left.StoreID == row.VectorStoreID && left.Namespace == namespace
			})
			if left, ok := s.leftBehind(stored, move.at); ok {
				pending = append(pending, left)
			}
		}
		encoded, err := encodeCleanups(pending)
		if err != nil {
			return err
		}
		row.VectorStoreCleanup = encoded
		return nil
	}
}

// clearIndexReturnedTo deletes, before a move back into a store, the index Warp left there
// under the namespace it is to use again, if upkeep has not deleted it yet: the log deletes
// made since reached only the store the index was in, so what it holds is no longer the
// logs', and a backfill rebuilds it.
func (s *Service) clearIndexReturnedTo(ctx context.Context, previous *tables.TableWarpConfig, target *vectorSelection, namespace string) error {
	store := target.store()
	if store == nil {
		return nil
	}
	for _, left := range pendingCleanups(previous) {
		if left.StoreID != target.entry.ID || left.Namespace != namespace {
			continue
		}
		if err := deleteWarpIndex(ctx, store, left.Namespace, left.Dimension); err != nil {
			return fmt.Errorf("delete the log index Warp left in vector store %q: %w", target.entry.Name, err)
		}
	}
	return nil
}

// targetVectorStore resolves a choice a save asks for, refusing one Warp could not use.
func (s *Service) targetVectorStore(name string) (*vectorSelection, error) {
	name = strings.TrimSpace(name)
	if s.vectorSlot == nil {
		if name != "" {
			return nil, fmt.Errorf("%w: vector_store %q: this deployment has no named vector stores to choose from", ErrInvalidConfig, name)
		}
		return &vectorSelection{}, nil
	}
	target := s.resolveSelection(name)
	if name != "" && target.entry == nil {
		return nil, fmt.Errorf("%w: vector_store %q is not configured in config.json", ErrInvalidConfig, name)
	}
	return target, nil
}

// LockBackfillAdmission takes the lock a backfill is admitted under, and returns its release.
// A start holds it from the in-flight lookup to the enqueue, and a save that moves the index
// holds it from its in-flight check to the write, so this process cannot admit a backfill for
// the store the index is leaving. A backfill admitted on another node checks its index on
// every page and stops at the first one after the move.
func (s *Service) LockBackfillAdmission() (unlock func()) {
	s.backfillAdmitMu.Lock()
	return s.backfillAdmitMu.Unlock
}

// selectionFollowingStore is the config store Warp's parts read through: every read of the
// configuration row makes Warp follow the store it chooses.
type selectionFollowingStore struct {
	configstore.WarpStore
	follow func(*tables.TableWarpConfig)
}

func (f *selectionFollowingStore) GetWarpConfig(ctx context.Context) (*tables.TableWarpConfig, error) {
	row, err := f.WarpStore.GetWarpConfig(ctx)
	if err == nil {
		f.follow(row)
	}
	return row, err
}
