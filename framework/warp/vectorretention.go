package warp

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/maximhq/bifrost/framework/configstore"
	"github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/maximhq/bifrost/framework/logstore"
	"github.com/maximhq/bifrost/framework/vectorstore"
)

// What the log store deletes, the vector store has to forget too, or a
// deleted request keeps turning up in semantic search and in topic runs
// with nothing behind its id. Two things delete logs: a person, through the
// Logs API, which the logging plugin announces to ForgetLogs; and retention,
// which is age, so the hourly upkeep sweeps vectors older than the vector
// retention (never longer than the logs') on its own clock. The assignment and pool rows go with the log
// inside the log store itself (see deleteWarpTopicRowsForLogs).
//
// The vector store is not in the log store's transaction, so a vector delete
// can fail after its log is gone. ForgetLogs records each one that does in the
// log store, and the hourly upkeep retries them until they go. Retention would
// not catch them: with none set nothing is swept, and a vector left behind
// keeps the deleted request's metadata - who sent it, its team, its cost - for
// as long as the deployment runs.

// SetVectorRetention tells the service how long each vector store keeps vectors,
// by the store's name ("" when Warp has no choice of stores), as a function so a
// retention changed at runtime is read at each sweep. The server passes the
// effective vector retention, which never exceeds the logs'. Zero or no function
// means no sweep.
func (s *Service) SetVectorRetention(days func(store string) int) {
	s.mu.Lock()
	s.vectorRetention = days
	s.mu.Unlock()
}

// vectorDeleteRetryBatch is how many pending vector deletes one upkeep pass
// retries.
const vectorDeleteRetryBatch = 500

// ForgetLogs removes the vectors of deleted logs. It never fails the delete
// that was asked for, which has already happened. A vector that is already
// gone is not an error; one it cannot remove - the vector store is down, or
// the request was cancelled - is recorded for upkeep to retry.
func (s *Service) ForgetLogs(ctx context.Context, ids []string) {
	if s.store == nil || len(ids) == 0 {
		return
	}
	// The stored namespace, whether or not Warp is switched on: a log deleted
	// while it is off still has a vector, and Config refuses a disabled Warp.
	// Without an embedding model nothing was ever indexed. A configuration
	// that cannot be read leaves every delete to the retry, which reads the
	// namespace then. The vector is in the store the configuration names: an
	// index Warp moved away from is deleted whole by upkeep.
	row, err := s.store.GetWarpConfig(ctx)
	if err != nil {
		failed := make([]logstore.WarpVectorDelete, 0, len(ids))
		for _, id := range ids {
			failed = append(failed, logstore.WarpVectorDelete{LogID: id})
		}
		s.recordVectorDeletes(ctx, failed, fmt.Errorf("read Warp configuration: %w", err))
		return
	}
	if row == nil || row.EmbeddingModel == "" {
		return
	}
	namespace := configFromRow(row).EffectiveLogVectorStoreNamespace()
	vectors := s.storeForRow(row)
	if vectors == nil {
		// A store that is configured but cannot be reached now still holds the
		// vectors, so the deletes wait for it. One no longer configured at all
		// cannot be reached by any later pass either.
		if unreachable := s.unreachableVectorStore(row); unreachable != nil {
			failed := make([]logstore.WarpVectorDelete, 0, len(ids))
			for _, id := range ids {
				failed = append(failed, logstore.WarpVectorDelete{LogID: id, Namespace: namespace})
			}
			s.recordVectorDeletes(ctx, failed, unreachable)
		}
		return
	}
	var failed []logstore.WarpVectorDelete
	for _, id := range ids {
		if err == nil {
			err = vectors.Delete(ctx, namespace, id)
			if err == nil || errors.Is(err, vectorstore.ErrNotFound) {
				err = nil
				continue
			}
		}
		// Once one fails the rest are recorded without trying: the store that
		// refused one, or the context that ended, refuses them all.
		failed = append(failed, logstore.WarpVectorDelete{LogID: id, Namespace: namespace})
	}
	if len(failed) == 0 {
		return
	}
	s.recordVectorDeletes(ctx, failed, err)
}

// unreachableVectorStore is why the store a row chooses cannot be used now when it is
// configured (it failed at startup, or is disabled); nil when it can be, or is not configured.
func (s *Service) unreachableVectorStore(row *tables.TableWarpConfig) error {
	if s.vectorSlot == nil || row == nil {
		return nil
	}
	selection := s.rowSelection(row)
	if selection.entry == nil || selection.store() != nil {
		return nil
	}
	return unavailableVectorStore(selection)
}

// recordVectorDeletes keeps the deletes ForgetLogs could not do, on a context
// of its own: the request's may be the reason they failed.
func (s *Service) recordVectorDeletes(ctx context.Context, failed []logstore.WarpVectorDelete, cause error) {
	store := s.topicStore()
	if store == nil {
		s.warnf("failed to remove the vectors of %d deleted log(s), and there is no log store to retry them from: %v", len(failed), cause)
		return
	}
	recordCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), topicsCleanupTimeout)
	defer cancel()
	if err := store.AddWarpVectorDeletes(recordCtx, failed); err != nil {
		s.warnf("failed to remove the vectors of %d deleted log(s) (%v), and could not record them to retry: %v", len(failed), cause, err)
		return
	}
	s.warnf("failed to remove the vectors of %d deleted log(s); topic upkeep will retry: %v", len(failed), cause)
}

// retryVectorDeletes retries the vector deletes ForgetLogs recorded, oldest
// first, and reports how many vectors are now gone. It stops at the first
// failure: the store that refused one is most likely refusing all of them,
// and the rest wait for the next pass.
func (s *Service) retryVectorDeletes(ctx context.Context) (int, error) {
	store := s.topicStore()
	if s.store == nil || store == nil {
		return 0, nil
	}
	// One read for the store and for the namespace of a delete recorded without
	// one, so the two always belong to the same configuration.
	row, err := s.store.GetWarpConfig(ctx)
	if err != nil || row == nil {
		return 0, err
	}
	vectors := s.storeForRow(row)
	if vectors == nil {
		return 0, nil
	}
	rows, err := store.ListWarpVectorDeletes(ctx, vectorDeleteRetryBatch)
	if err != nil || len(rows) == 0 {
		return 0, err
	}
	configured := configFromRow(row).EffectiveLogVectorStoreNamespace()
	done := make([]string, 0, len(rows))
	// A ClickHouse table lists a log recorded twice twice until it merges.
	seen := make(map[string]struct{}, len(rows))
	var failure error
	for _, row := range rows {
		if _, repeat := seen[row.LogID]; repeat {
			continue
		}
		seen[row.LogID] = struct{}{}
		namespace := row.Namespace
		if namespace == "" {
			namespace = configured
		}
		if err := vectors.Delete(ctx, namespace, row.LogID); err != nil && !errors.Is(err, vectorstore.ErrNotFound) {
			failure = err
			break
		}
		done = append(done, row.LogID)
	}
	if len(done) > 0 {
		if err := store.DeleteWarpVectorDeletes(ctx, done); err != nil {
			return 0, fmt.Errorf("clear retried vector deletes: %w", err)
		}
	}
	if failure != nil {
		return len(done), fmt.Errorf("retry vector delete: %w", failure)
	}
	return len(done), nil
}

// retentionCutoff is now less a retention of days. A retention too long for a
// time.Duration would wrap into a cutoff after now and sweep everything, so it
// is held at the longest that fits, which keeps every vector a real
// deployment has. Validation bounds retention_days the same way; this
// also covers a long log retention, which the effective retention follows.
func retentionCutoff(now time.Time, days int) time.Time {
	return now.Add(-time.Duration(min(days, configstore.MaxVectorRetentionDays)) * 24 * time.Hour)
}

// sweepExpiredVectors deletes Warp's vectors for logs older than the vector
// retention, and reports how many went.
func (s *Service) sweepExpiredVectors(ctx context.Context, now time.Time) (int, error) {
	s.mu.RLock()
	retention := s.vectorRetention
	s.mu.RUnlock()
	if retention == nil || s.store == nil {
		return 0, nil
	}
	// The stored namespace, whether or not Warp is switched on, as ForgetLogs
	// reads it: logs keep expiring while Warp is off. Without an embedding
	// model nothing was ever indexed, so there is nothing to sweep. The index
	// is swept in the store the configuration names, under that store's
	// retention.
	row, err := s.store.GetWarpConfig(ctx)
	if err != nil || row == nil || row.EmbeddingModel == "" {
		return 0, err
	}
	vectors := s.storeForRow(row)
	if vectors == nil {
		return 0, nil
	}
	days := retention(s.vectorStoreName(row))
	if days <= 0 {
		return 0, nil
	}
	cutoff := retentionCutoff(now, days)
	deleted, err := vectors.DeleteAll(ctx, configFromRow(row).EffectiveLogVectorStoreNamespace(), []vectorstore.Query{
		{Field: "warp_log", Operator: vectorstore.QueryOperatorEqual, Value: true},
		{Field: "timestamp", Operator: vectorstore.QueryOperatorLessThan, Value: cutoff.Unix()},
	})
	return len(deleted), err
}

// vectorStoreName is the name of the store a row chooses, for its settings; "" without a
// choice of stores.
func (s *Service) vectorStoreName(row *tables.TableWarpConfig) string {
	if entry := s.rowSelection(row).entry; entry != nil {
		return entry.Name
	}
	return ""
}

// VectorStorageLimitIndexes is Warp's log index, for the vector storage limit, which bounds
// the space it takes together with the other log-derived indexes and deletes
// their oldest vectors first. None without a vector store or an embedding
// model (nothing was indexed), or while the configuration cannot be read.
func (s *Service) VectorStorageLimitIndexes(ctx context.Context) []vectorstore.StorageLimitIndex {
	if s.store == nil {
		return nil
	}
	row, err := s.store.GetWarpConfig(ctx)
	if err != nil || row == nil || row.EmbeddingModel == "" {
		return nil
	}
	vectors := s.storeForRow(row)
	if vectors == nil {
		return nil
	}
	config := configFromRow(row)
	namespace := config.EffectiveLogVectorStoreNamespace()
	// Opened before it is counted, with the width the configuration gives: an
	// embedded Chromem store cannot enumerate a namespace this process has not
	// opened, and would otherwise drop out of the limit after a restart until
	// the next log was indexed. The indexer opens it once per configuration. If
	// opening fails, the index is still offered, and the limit reports it as
	// skipped with the reason.
	if s.indexer != nil {
		if err := s.indexer.ensureNamespaceOnce(ctx, vectors, namespace, config.EmbeddingDimension); err != nil {
			s.warnf("failed to open Warp's log index for the vector storage limit: %v", err)
		}
	}
	// The store itself, not the Switchable over it, so the limit of the store it
	// is in counts it.
	return []vectorstore.StorageLimitIndex{{
		Name: "Warp's log index", Store: vectors, Namespace: namespace,
		Dimension: config.EmbeddingDimension, TimestampField: "timestamp",
		Filter: []vectorstore.Query{{Field: "warp_log", Operator: vectorstore.QueryOperatorEqual, Value: true}},
	}}
}
