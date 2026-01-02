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

"""Tests for stream module."""

import time
from unittest.mock import MagicMock

from custom_components.cardata.stream import CardataStreamManager, ConnectionState


class TestConnectionState:
    """Tests for ConnectionState enum."""

    def test_states_defined(self):
        """Test all expected states are defined."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.DISCONNECTING.value == "disconnecting"
        assert ConnectionState.FAILED.value == "failed"

    def test_states_count(self):
        """Test we have exactly 5 states."""
        assert len(ConnectionState) == 5


class TestCardataStreamManagerCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def create_manager(self):
        """Create a CardataStreamManager with mocked dependencies."""
        mock_hass = MagicMock()
        mock_hass.loop = MagicMock()
        return CardataStreamManager(
            hass=mock_hass,
            client_id="test_client",
            gcid="test_gcid",
            id_token="test_token",
            host="test.host.com",
            port=8883,
            keepalive=60,
        )

    def test_initial_circuit_breaker_state(self):
        """Test circuit breaker starts in closed state."""
        manager = self.create_manager()
        assert manager._circuit_open is False
        assert manager._circuit_open_until is None
        assert manager._failure_count == 0

    def test_record_failure_increments_count(self):
        """Test recording failures increments the failure count."""
        manager = self.create_manager()

        manager._record_failure()
        assert manager._failure_count == 1

        manager._record_failure()
        assert manager._failure_count == 2

    def test_circuit_opens_after_max_failures(self):
        """Test circuit breaker opens after reaching max failures."""
        manager = self.create_manager()
        manager._max_failures_per_window = 3  # Lower threshold for testing

        for _ in range(3):
            manager._record_failure()

        assert manager._circuit_open is True
        assert manager._circuit_open_until is not None

    def test_check_circuit_breaker_when_open(self):
        """Test check_circuit_breaker returns True when open."""
        manager = self.create_manager()
        manager._circuit_open = True
        manager._circuit_open_until = time.monotonic() + 300  # 5 minutes from now

        result = manager._check_circuit_breaker()

        assert result is True

    def test_check_circuit_breaker_when_closed(self):
        """Test check_circuit_breaker returns False when closed."""
        manager = self.create_manager()
        manager._circuit_open = False

        result = manager._check_circuit_breaker()

        assert result is False

    def test_circuit_breaker_resets_after_timeout(self):
        """Test circuit breaker resets when timeout expires."""
        manager = self.create_manager()
        manager._circuit_open = True
        manager._circuit_open_until = time.monotonic() - 1  # Already expired

        result = manager._check_circuit_breaker()

        assert result is False
        assert manager._circuit_open is False
        assert manager._circuit_open_until is None
        assert manager._failure_count == 0

    def test_record_success_resets_circuit_breaker(self):
        """Test successful connection resets circuit breaker."""
        manager = self.create_manager()
        manager._circuit_open = True
        manager._circuit_open_until = time.monotonic() + 300
        manager._failure_count = 10

        manager._record_success()

        assert manager._circuit_open is False
        assert manager._circuit_open_until is None
        assert manager._failure_count == 0

    def test_failure_window_resets_after_timeout(self):
        """Test failure window resets after timeout expires."""
        manager = self.create_manager()
        manager._failure_window_seconds = 60
        manager._failure_window_start = time.monotonic() - 120  # 2 minutes ago
        manager._failure_count = 5

        manager._check_circuit_breaker()

        assert manager._failure_count == 0
        assert manager._failure_window_start is None

    def test_get_circuit_breaker_state_when_closed(self):
        """Test get_circuit_breaker_state when circuit is closed."""
        manager = self.create_manager()

        state = manager.get_circuit_breaker_state()

        assert state == {"circuit_open": False}

    def test_get_circuit_breaker_state_when_open(self):
        """Test get_circuit_breaker_state when circuit is open."""
        manager = self.create_manager()
        manager._circuit_open = True
        manager._circuit_open_until = time.monotonic() + 300
        manager._failure_count = 10

        state = manager.get_circuit_breaker_state()

        assert state["circuit_open"] is True
        assert "circuit_open_until" in state
        assert state["failure_count"] == 10

    def test_restore_circuit_breaker_state_empty(self):
        """Test restoring empty state does nothing."""
        manager = self.create_manager()

        manager.restore_circuit_breaker_state({})

        assert manager._circuit_open is False

    def test_restore_circuit_breaker_state_closed(self):
        """Test restoring closed state does nothing."""
        manager = self.create_manager()

        manager.restore_circuit_breaker_state({"circuit_open": False})

        assert manager._circuit_open is False

    def test_restore_circuit_breaker_state_expired(self):
        """Test restoring expired state doesn't open circuit."""
        manager = self.create_manager()

        # Expired in the past
        manager.restore_circuit_breaker_state(
            {
                "circuit_open": True,
                "circuit_open_until": time.time() - 100,
                "failure_count": 10,
            }
        )

        assert manager._circuit_open is False

    def test_restore_circuit_breaker_state_active(self):
        """Test restoring active circuit breaker state."""
        manager = self.create_manager()

        # 5 minutes from now
        future_time = time.time() + 300

        manager.restore_circuit_breaker_state(
            {
                "circuit_open": True,
                "circuit_open_until": future_time,
                "failure_count": 10,
            }
        )

        assert manager._circuit_open is True
        assert manager._circuit_open_until is not None
        assert manager._failure_count == 10


class TestCardataStreamManagerDebugInfo:
    """Tests for debug info functionality."""

    def create_manager(self):
        """Create a CardataStreamManager with mocked dependencies."""
        mock_hass = MagicMock()
        mock_hass.loop = MagicMock()
        return CardataStreamManager(
            hass=mock_hass,
            client_id="test_client",
            gcid="test_gcid",
            id_token="test_token_1234567890_secret",
            host="test.host.com",
            port=8883,
            keepalive=60,
        )

    def test_debug_info_contains_expected_fields(self):
        """Test debug_info contains all expected fields."""
        manager = self.create_manager()

        info = manager.debug_info

        assert "client_id" in info
        assert "gcid" in info
        assert "host" in info
        assert "port" in info
        assert "keepalive" in info
        assert "topic" in info
        assert "clean_session" in info
        assert "protocol" in info
        assert "id_token" in info

    def test_debug_info_redacts_token(self):
        """Test debug_info redacts the token."""
        manager = self.create_manager()

        info = manager.debug_info

        # Token should be redacted to first 10 chars + ...
        assert info["id_token"] == "test_token..."
        # Should not contain full token
        assert "secret" not in info["id_token"]

    def test_debug_info_values(self):
        """Test debug_info contains correct values."""
        manager = self.create_manager()

        info = manager.debug_info

        assert info["client_id"] == "test_client"
        assert info["gcid"] == "test_gcid"
        assert info["host"] == "test.host.com"
        assert info["port"] == 8883
        assert info["keepalive"] == 60
        assert info["topic"] == "test_gcid/+"
        assert info["clean_session"] is True
        assert info["protocol"] == "MQTTv311"


class TestCardataStreamManagerCallbacks:
    """Tests for callback setting functionality."""

    def create_manager(self):
        """Create a CardataStreamManager with mocked dependencies."""
        mock_hass = MagicMock()
        mock_hass.loop = MagicMock()
        return CardataStreamManager(
            hass=mock_hass,
            client_id="test_client",
            gcid="test_gcid",
            id_token="test_token",
            host="test.host.com",
            port=8883,
            keepalive=60,
        )

    def test_set_message_callback(self):
        """Test setting message callback."""
        manager = self.create_manager()
        callback = MagicMock()

        manager.set_message_callback(callback)

        assert manager._message_callback == callback

    def test_set_status_callback(self):
        """Test setting status callback."""
        manager = self.create_manager()
        callback = MagicMock()

        manager.set_status_callback(callback)

        assert manager._status_callback == callback

    def test_client_property_initially_none(self):
        """Test client property is initially None."""
        manager = self.create_manager()

        assert manager.client is None
