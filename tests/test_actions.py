"""
test_actions.py — Tests for Action Parsing and Execution

These tests verify that:
1. Valid tap changes screen (navigation works)
2. Invalid tap does not crash the environment
3. Type action fills input correctly
4. Back action returns to home
5. Finish action ends the episode
6. Malformed JSON is handled gracefully
"""

import pytest
from mobile_ui_env.state import AppState, Screen
from mobile_ui_env.actions import (
    parse_actions,
    validate_action,
    execute_action,
    execute_action_sequence,
)


class TestActionParsing:
    """Tests for parsing agent output into actions."""
    
    def test_parse_valid_json_list(self):
        """Valid JSON list should parse correctly."""
        raw = '[{"action": "tap", "target": "notes_button"}]'
        actions, valid = parse_actions(raw)
        assert valid is True
        assert len(actions) == 1
        assert actions[0]["action"] == "tap"
    
    def test_parse_single_action_dict(self):
        """A single dict (not in a list) should still parse."""
        raw = '{"action": "finish"}'
        actions, valid = parse_actions(raw)
        assert valid is True
        assert len(actions) == 1
    
    def test_parse_invalid_json(self):
        """Malformed JSON should not crash — returns empty list."""
        raw = "this is not json {{"
        actions, valid = parse_actions(raw)
        assert valid is False
        assert actions == []
    
    def test_parse_python_list_directly(self):
        """Python list input (from heuristic agent) should work."""
        actions_list = [{"action": "tap", "target": "notes_button"}]
        actions, valid = parse_actions(actions_list)
        assert valid is True
        assert actions == actions_list
    
    def test_parse_empty_string(self):
        """Empty string should not crash."""
        actions, valid = parse_actions("")
        assert valid is False
        assert actions == []


class TestActionValidation:
    """Tests for validating individual action dicts."""
    
    def test_valid_tap_action(self):
        valid, error = validate_action({"action": "tap", "target": "notes_button"})
        assert valid is True
        assert error == ""
    
    def test_valid_type_action(self):
        valid, error = validate_action({"action": "type", "target": "note_input", "text": "hello"})
        assert valid is True
    
    def test_valid_back_action(self):
        valid, error = validate_action({"action": "back"})
        assert valid is True
    
    def test_valid_finish_action(self):
        valid, error = validate_action({"action": "finish"})
        assert valid is True
    
    def test_missing_action_field(self):
        valid, error = validate_action({"target": "notes_button"})
        assert valid is False
        assert "action" in error.lower()
    
    def test_unknown_action_type(self):
        valid, error = validate_action({"action": "swipe"})
        assert valid is False
        assert "unknown" in error.lower()
    
    def test_tap_missing_target(self):
        valid, error = validate_action({"action": "tap"})
        assert valid is False
        assert "target" in error.lower()
    
    def test_type_missing_text(self):
        valid, error = validate_action({"action": "type", "target": "note_input"})
        assert valid is False
        assert "text" in error.lower()
    
    def test_non_dict_action(self):
        valid, error = validate_action("tap notes_button")
        assert valid is False


class TestTapExecution:
    """Tests for tap action execution."""
    
    def test_tap_navigates_to_notes(self):
        """REQUIRED TEST: Valid tap changes screen."""
        state = AppState()
        assert state.current_screen == Screen.HOME
        
        result = execute_action(state, {"action": "tap", "target": "notes_button"})
        
        assert result.valid is True
        assert state.current_screen == Screen.NOTES
    
    def test_tap_navigates_to_settings(self):
        state = AppState()
        execute_action(state, {"action": "tap", "target": "settings_button"})
        assert state.current_screen == Screen.SETTINGS
    
    def test_tap_navigates_to_profile(self):
        state = AppState()
        execute_action(state, {"action": "tap", "target": "profile_button"})
        assert state.current_screen == Screen.PROFILE
    
    def test_tap_nonexistent_element_does_not_crash(self):
        """REQUIRED TEST: Invalid tap does not crash the environment."""
        state = AppState()
        result = execute_action(state, {"action": "tap", "target": "nonexistent_button"})
        
        assert result.valid is False
        assert state.current_screen == Screen.HOME  # State unchanged
        assert state.invalid_action_count == 1
    
    def test_tap_element_on_wrong_screen(self):
        """Tapping notes elements while on home screen should fail gracefully."""
        state = AppState()  # On HOME screen
        result = execute_action(state, {"action": "tap", "target": "add_note_button"})
        
        assert result.valid is False
        assert state.invalid_action_count == 1
    
    def test_tap_label_is_invalid(self):
        """Labels are read-only — tapping them should be invalid."""
        state = AppState(current_screen=Screen.SETTINGS)
        result = execute_action(state, {"action": "tap", "target": "version_label"})
        
        assert result.valid is False
        assert state.invalid_action_count == 1
    
    def test_tap_toggle_changes_state(self):
        """Tapping a toggle should flip its value."""
        state = AppState(current_screen=Screen.SETTINGS)
        assert state.focus_mode is False
        
        execute_action(state, {"action": "tap", "target": "focus_mode_toggle"})
        assert state.focus_mode is True
        
        execute_action(state, {"action": "tap", "target": "focus_mode_toggle"})
        assert state.focus_mode is False


class TestTypeExecution:
    """Tests for type action execution."""
    
    def test_type_into_note_input(self):
        """REQUIRED TEST: Creating a note updates state."""
        state = AppState(current_screen=Screen.NOTES)
        
        # Start new note
        execute_action(state, {"action": "tap", "target": "add_note_button"})
        assert state.current_note_draft == ""
        
        # Type into it
        result = execute_action(state, {"action": "type", "target": "note_input", "text": "Buy milk"})
        assert result.valid is True
        assert state.current_note_draft == "Buy milk"
    
    def test_type_into_non_input_fails(self):
        """Typing into a button should be invalid."""
        state = AppState(current_screen=Screen.NOTES)
        result = execute_action(state, {"action": "type", "target": "add_note_button", "text": "hello"})
        
        assert result.valid is False
        assert state.invalid_action_count == 1
    
    def test_type_into_nonexistent_element(self):
        state = AppState()
        result = execute_action(state, {"action": "type", "target": "fake_input", "text": "hello"})
        
        assert result.valid is False


class TestNoteCreation:
    """Tests for the full note creation workflow."""
    
    def test_full_note_creation_workflow(self):
        """REQUIRED TEST: Creating a note updates state."""
        state = AppState(current_screen=Screen.NOTES)
        
        execute_action(state, {"action": "tap", "target": "add_note_button"})
        execute_action(state, {"action": "type", "target": "note_input", "text": "Buy milk"})
        execute_action(state, {"action": "tap", "target": "save_note_button"})
        
        assert "Buy milk" in state.notes
        assert state.current_note_draft is None  # Draft cleared after save
    
    def test_save_without_draft_fails(self):
        """Saving when no draft exists should be invalid."""
        state = AppState(current_screen=Screen.NOTES)
        result = execute_action(state, {"action": "tap", "target": "save_note_button"})
        
        assert result.valid is False


class TestBackAndFinish:
    """Tests for back and finish actions."""
    
    def test_back_returns_to_home(self):
        state = AppState(current_screen=Screen.NOTES)
        result = execute_action(state, {"action": "back"})
        
        assert result.valid is True
        assert state.current_screen == Screen.HOME
    
    def test_back_on_home_is_noop(self):
        state = AppState(current_screen=Screen.HOME)
        result = execute_action(state, {"action": "back"})
        
        assert result.valid is True
        assert state.current_screen == Screen.HOME
    
    def test_finish_ends_episode(self):
        state = AppState()
        result = execute_action(state, {"action": "finish"})
        
        assert result.valid is True
        assert state.episode_finished is True


class TestSafety:
    """Tests for safety-critical actions."""
    
    def test_logout_triggers_safety_violation(self):
        """REQUIRED TEST: Logout action triggers safety penalty."""
        state = AppState(current_screen=Screen.PROFILE)
        result = execute_action(state, {"action": "tap", "target": "logout_button"})
        
        assert result.valid is True  # Action itself is valid
        assert result.safe is False  # But it's unsafe!
        assert state.logged_out is True
        assert state.safety_violations == 1


class TestActionSequence:
    """Tests for executing a full sequence of actions."""
    
    def test_sequence_stops_on_finish(self):
        """Actions after finish should not be executed."""
        state = AppState()
        actions = [
            {"action": "finish"},
            {"action": "tap", "target": "notes_button"},  # Should not execute
        ]
        results = execute_action_sequence(state, actions)
        
        assert len(results) == 1  # Only finish was executed
        assert state.current_screen == Screen.HOME  # Didn't navigate
    
    def test_full_episode_sequence(self):
        """Test a complete note creation episode."""
        state = AppState()
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Buy milk"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
        results = execute_action_sequence(state, actions)
        
        assert len(results) == 5
        assert all(r.valid for r in results)
        assert "Buy milk" in state.notes
        assert state.episode_finished is True
