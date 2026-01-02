# Copyright (c) 2025, Renaud Allard <renaud@allard.it>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Tests for pending_manager module."""

from datetime import datetime, timedelta

from custom_components.cardata.pending_manager import PendingManager, PendingSnapshot


class TestPendingManagerBasic:
    """Tests for basic PendingManager operations."""

    def test_add_update_single(self):
        """Test adding a single update."""
        pm = PendingManager()
        assert pm.add_update("VIN123", "descriptor.a") is True
        assert pm.get_total_count() == 1
        assert pm.has_pending() is True

    def test_add_update_multiple_same_vin(self):
        """Test adding multiple updates for same VIN."""
        pm = PendingManager()
        pm.add_update("VIN123", "descriptor.a")
        pm.add_update("VIN123", "descriptor.b")
        pm.add_update("VIN123", "descriptor.c")
        assert pm.get_total_count() == 3

    def test_add_update_duplicate(self):
        """Test adding duplicate descriptor (set deduplication)."""
        pm = PendingManager()
        pm.add_update("VIN123", "descriptor.a")
        pm.add_update("VIN123", "descriptor.a")
        assert pm.get_total_count() == 1

    def test_add_new_sensor(self):
        """Test adding new sensor notification."""
        pm = PendingManager()
        assert pm.add_new_sensor("VIN123", "sensor.temp") is True
        assert pm.get_total_count() == 1

    def test_add_new_binary(self):
        """Test adding new binary sensor notification."""
        pm = PendingManager()
        assert pm.add_new_binary("VIN123", "binary.door") is True
        assert pm.get_total_count() == 1

    def test_multiple_types_combined(self):
        """Test combining updates, sensors, and binary sensors."""
        pm = PendingManager()
        pm.add_update("VIN123", "descriptor.a")
        pm.add_new_sensor("VIN123", "sensor.temp")
        pm.add_new_binary("VIN123", "binary.door")
        assert pm.get_total_count() == 3

    def test_started_at_tracked(self):
        """Test that started_at is set on first add."""
        pm = PendingManager()
        assert pm.started_at is None
        pm.add_update("VIN123", "descriptor.a")
        assert pm.started_at is not None

    def test_has_pending_empty(self):
        """Test has_pending returns False when empty."""
        pm = PendingManager()
        assert pm.has_pending() is False


class TestPendingManagerSnapshot:
    """Tests for snapshot_and_clear functionality."""

    def test_snapshot_returns_data(self):
        """Test that snapshot contains the correct data."""
        pm = PendingManager()
        pm.add_update("VIN123", "descriptor.a")
        pm.add_update("VIN456", "descriptor.b")
        pm.add_new_sensor("VIN123", "sensor.temp")
        pm.add_new_binary("VIN456", "binary.door")

        snapshot = pm.snapshot_and_clear()

        assert isinstance(snapshot, PendingSnapshot)
        assert "VIN123" in snapshot.updates
        assert "descriptor.a" in snapshot.updates["VIN123"]
        assert "VIN456" in snapshot.updates
        assert "VIN123" in snapshot.new_sensors
        assert "VIN456" in snapshot.new_binary

    def test_snapshot_clears_state(self):
        """Test that snapshot clears all internal state."""
        pm = PendingManager()
        pm.add_update("VIN123", "descriptor.a")
        pm.add_new_sensor("VIN123", "sensor.temp")
        pm.add_new_binary("VIN123", "binary.door")

        pm.snapshot_and_clear()

        assert pm.get_total_count() == 0
        assert pm.has_pending() is False
        assert pm.started_at is None

    def test_snapshot_empty(self):
        """Test snapshot on empty manager."""
        pm = PendingManager()
        snapshot = pm.snapshot_and_clear()
        assert len(snapshot.updates) == 0
        assert len(snapshot.new_sensors) == 0
        assert len(snapshot.new_binary) == 0


class TestPendingManagerRemove:
    """Tests for remove_vin functionality."""

    def test_remove_vin_clears_all_types(self):
        """Test removing a VIN clears all data types for that VIN."""
        pm = PendingManager()
        pm.add_update("VIN123", "descriptor.a")
        pm.add_new_sensor("VIN123", "sensor.temp")
        pm.add_new_binary("VIN123", "binary.door")
        pm.add_update("VIN456", "descriptor.b")

        pm.remove_vin("VIN123")

        assert pm.get_total_count() == 1  # Only VIN456 descriptor left

    def test_remove_nonexistent_vin(self):
        """Test removing non-existent VIN doesn't raise."""
        pm = PendingManager()
        pm.add_update("VIN123", "descriptor.a")
        pm.remove_vin("NONEXISTENT")  # Should not raise
        assert pm.get_total_count() == 1


class TestPendingManagerStaleClearing:
    """Tests for stale update clearing."""

    def test_clear_stale_when_not_stale(self):
        """Test that non-stale updates are not cleared."""
        pm = PendingManager()
        pm.add_update("VIN123", "descriptor.a")

        now = datetime.now()
        cleared = pm.check_and_clear_stale(now)

        assert cleared == 0
        assert pm.get_total_count() == 1

    def test_clear_stale_when_stale(self):
        """Test that stale updates are cleared."""
        pm = PendingManager()
        pm.add_update("VIN123", "descriptor.a")
        pm.add_update("VIN123", "descriptor.b")

        # Simulate old started_at
        pm._started_at = datetime.now() - timedelta(seconds=pm.MAX_AGE_SECONDS + 10)

        now = datetime.now()
        cleared = pm.check_and_clear_stale(now)

        assert cleared == 2
        assert pm.get_total_count() == 0
        assert pm.started_at is None

    def test_clear_stale_no_pending(self):
        """Test clearing stale when nothing pending."""
        pm = PendingManager()
        now = datetime.now()
        cleared = pm.check_and_clear_stale(now)
        assert cleared == 0


class TestPendingManagerEviction:
    """Tests for eviction logic."""

    def test_evict_updates(self):
        """Test evict_updates removes half from largest VIN."""
        pm = PendingManager()
        # Add many updates to one VIN
        for i in range(10):
            pm.add_update("VIN123", f"descriptor.{i}")

        evicted = pm.evict_updates()

        assert evicted == 5  # Half of 10
        # Remaining count should be reduced
        assert pm.get_total_count() == 5

    def test_evict_updates_empty(self):
        """Test evict_updates on empty manager."""
        pm = PendingManager()
        evicted = pm.evict_updates()
        assert evicted == 0

    def test_evict_vin(self):
        """Test evict_vin removes VIN with fewest updates."""
        pm = PendingManager()
        pm.add_update("VIN_SMALL", "descriptor.a")
        for i in range(10):
            pm.add_update("VIN_LARGE", f"descriptor.{i}")

        evicted = pm.evict_vin()

        assert evicted == 1  # VIN_SMALL had 1 item
        # VIN_LARGE should still exist
        assert pm.get_total_count() == 10

    def test_evict_vin_empty(self):
        """Test evict_vin on empty manager."""
        pm = PendingManager()
        evicted = pm.evict_vin()
        assert evicted == 0

    def test_per_vin_limit_triggers_eviction(self):
        """Test that per-VIN limit triggers eviction."""
        pm = PendingManager()
        pm.MAX_PER_VIN = 10  # Lower limit for testing

        # Add up to the limit
        for i in range(10):
            pm.add_update("VIN123", f"descriptor.{i}")

        # Adding one more should trigger eviction
        pm.add_update("VIN123", "descriptor.new")

        # Should have evicted half (5) and added 1
        assert pm.get_total_count() <= 10

    def test_max_total_limit(self):
        """Test that MAX_TOTAL limit is enforced."""
        pm = PendingManager()
        pm.MAX_TOTAL = 20  # Lower limit for testing

        # Try to add more than max
        for i in range(25):
            pm.add_update("VIN123", f"descriptor.{i}")

        # Should be at or below MAX_TOTAL
        assert pm.get_total_count() <= pm.MAX_TOTAL

    def test_max_vins_limit(self):
        """Test that MAX_VINS limit is enforced."""
        pm = PendingManager()
        pm.MAX_VINS = 3  # Lower limit for testing

        # Add updates for 3 VINs
        for i in range(3):
            pm.add_update(f"VIN{i}", "descriptor.a")

        # Adding a 4th VIN should trigger eviction
        pm.add_update("VIN_NEW", "descriptor.a")

        # Count unique VINs
        snapshot = pm.snapshot_and_clear()
        all_vins = set(snapshot.updates.keys())
        assert len(all_vins) <= 3


class TestPendingManagerEvictedCount:
    """Tests for evicted_count tracking."""

    def test_evicted_count_increments(self):
        """Test that evicted_count tracks evictions."""
        pm = PendingManager()
        pm.MAX_TOTAL = 10  # Lower limit for testing

        assert pm.evicted_count == 0

        # Force eviction by exceeding limits
        for i in range(15):
            pm.add_update("VIN123", f"descriptor.{i}")

        # Should have evicted some
        assert pm.evicted_count > 0


class TestPendingSnapshot:
    """Tests for PendingSnapshot namedtuple."""

    def test_snapshot_is_namedtuple(self):
        """Test PendingSnapshot is a proper NamedTuple."""
        snapshot = PendingSnapshot(
            updates={"VIN1": {"desc.a"}},
            new_sensors={"VIN2": {"sensor.a"}},
            new_binary={},
        )
        assert snapshot.updates == {"VIN1": {"desc.a"}}
        assert snapshot.new_sensors == {"VIN2": {"sensor.a"}}
        assert snapshot.new_binary == {}

    def test_snapshot_unpacking(self):
        """Test PendingSnapshot can be unpacked."""
        snapshot = PendingSnapshot(
            updates={"VIN1": {"desc.a"}},
            new_sensors={},
            new_binary={},
        )
        updates, new_sensors, new_binary = snapshot
        assert updates == {"VIN1": {"desc.a"}}
        assert new_sensors == {}
        assert new_binary == {}
