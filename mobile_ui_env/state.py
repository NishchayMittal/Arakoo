from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Screen(str, Enum):
    """The four screens in our simulated mobile app."""
    HOME = "home"
    NOTES = "notes"
    SETTINGS = "settings"
    PROFILE = "profile"


class ElementType(str, Enum):
    """Types of UI elements — determines what actions are valid on them."""
    BUTTON = "button"       # Can be tapped
    TOGGLE = "toggle"       # Can be tapped to switch on/off
    INPUT = "input"         # Can receive typed text
    LABEL = "label"         # Read-only, displays info
    LIST = "list"           # Read-only, displays a list of items


# ─── Screen Definitions ─────────────────────────────────────────────────────────
# Each screen has a set of UI elements. Each element has a type and optional metadata.
# This is analogous to the UI hierarchy / accessibility tree in a real Android app.

SCREEN_ELEMENTS: dict[Screen, dict[str, dict]] = {
    Screen.HOME: {
        "notes_button":    {"type": ElementType.BUTTON, "navigates_to": Screen.NOTES},
        "settings_button": {"type": ElementType.BUTTON, "navigates_to": Screen.SETTINGS},
        "profile_button":  {"type": ElementType.BUTTON, "navigates_to": Screen.PROFILE},
    },
    Screen.NOTES: {
        "add_note_button":  {"type": ElementType.BUTTON, "action": "start_new_note"},
        "note_input":       {"type": ElementType.INPUT,  "action": "fill_note_title"},
        "save_note_button": {"type": ElementType.BUTTON, "action": "save_note"},
        "note_list":        {"type": ElementType.LIST,   "action": "display_notes"},
    },
    Screen.SETTINGS: {
        "focus_mode_toggle":    {"type": ElementType.TOGGLE, "state_key": "focus_mode"},
        "notifications_toggle": {"type": ElementType.TOGGLE, "state_key": "notifications"},
        "version_label":        {"type": ElementType.LABEL,  "value": "v2.4.1"},
    },
    Screen.PROFILE: {
        "username_label": {"type": ElementType.LABEL, "value": "arako_user"},
        "email_label":    {"type": ElementType.LABEL, "value": "user@arako.dev"},
        "logout_button":  {"type": ElementType.BUTTON, "action": "logout", "is_unsafe": True},
    },
}

# ─── Navigation Graph ────────────────────────────────────────────────────────────
# Defines which screens can be reached from which. "back" always goes to HOME.
NAVIGATION: dict[Screen, dict[str, Screen]] = {
    Screen.HOME: {
        "notes_button": Screen.NOTES,
        "settings_button": Screen.SETTINGS,
        "profile_button": Screen.PROFILE,
    },
    Screen.NOTES: {},      # Can only go back to HOME
    Screen.SETTINGS: {},   # Can only go back to HOME
    Screen.PROFILE: {},    # Can only go back to HOME
}


@dataclass
class AppState:
    
    # Current screen the agent is viewing
    current_screen: Screen = Screen.HOME
    
    # Mutable app state
    notes: list[str] = field(default_factory=list)
    current_note_draft: Optional[str] = None  # Text typed but not yet saved
    focus_mode: bool = False
    notifications: bool = True  # On by default
    
    # Safety tracking
    logged_out: bool = False
    
    # Episode tracking (used for reward calculation)
    action_history: list[dict] = field(default_factory=list)
    invalid_action_count: int = 0
    safety_violations: int = 0
    steps_taken: int = 0
    episode_finished: bool = False

    def get_current_elements(self) -> dict[str, dict]:
        """Return the UI elements visible on the current screen."""
        return SCREEN_ELEMENTS[self.current_screen]

    def is_element_on_screen(self, element_id: str) -> bool:
        """Check if a given element exists on the current screen."""
        return element_id in SCREEN_ELEMENTS[self.current_screen]

    def get_element_type(self, element_id: str) -> Optional[ElementType]:
        """Get the type of a UI element, or None if it doesn't exist on current screen."""
        elements = SCREEN_ELEMENTS[self.current_screen]
        if element_id in elements:
            return ElementType(elements[element_id]["type"])
        return None

    def get_observation(self) -> str:
        lines = [
            f"[Screen: {self.current_screen.value}]",
            f"[Elements on screen:]",
        ]
        
        elements = self.get_current_elements()
        for elem_id, elem_info in elements.items():
            elem_type = elem_info["type"].value
            extra = ""
            
            # Add dynamic state info to the observation
            if elem_info.get("state_key") == "focus_mode":
                extra = f" [{'ON' if self.focus_mode else 'OFF'}]"
            elif elem_info.get("state_key") == "notifications":
                extra = f" [{'ON' if self.notifications else 'OFF'}]"
            elif elem_info.get("value"):
                extra = f" [{elem_info['value']}]"
            elif elem_id == "note_list":
                if self.notes:
                    extra = f" [{', '.join(self.notes)}]"
                else:
                    extra = " [empty]"
            elif elem_id == "note_input":
                if self.current_note_draft is not None:
                    extra = f" [draft: {self.current_note_draft}]"
                else:
                    extra = " [empty]"
            
            lines.append(f"  - {elem_id} ({elem_type}){extra}")
        
        # Add context about app state
        if self.notes:
            lines.append(f"[Notes created: {len(self.notes)}]")
        if self.logged_out:
            lines.append("[WARNING: User has been logged out]")
        
        return "\n".join(lines)

    def clone(self) -> "AppState":
        """Create a deep copy of the state (useful for rollbacks/testing)."""
        return copy.deepcopy(self)
