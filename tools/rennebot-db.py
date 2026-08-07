#!/usr/bin/env python3
"""Administrate RenneBot runtime settings directly in its SQLite database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SETTING_ALIASES = {
    "admins": "admin_user_ids",
    "ai-users": "ai_private_user_ids",
    "ai-groups": "ai_group_ids",
}


def main() -> None:
    """Parse a local administrator command and update the selected SQLite file."""
    from qq_game_registry.database import PluginDatabase

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, required=True, help="Path to rennebot.sqlite3"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "show", help="Show configured administrators and AI allowlists"
    )

    set_parser = subparsers.add_parser("set", help="Replace one ID-list setting")
    set_parser.add_argument("setting", choices=SETTING_ALIASES)
    set_parser.add_argument("ids", nargs="+", help="One or more platform IDs")

    json_parser = subparsers.add_parser("set-json", help="Set any JSON plugin setting")
    json_parser.add_argument("key")
    json_parser.add_argument("value", help="Valid JSON value")

    arguments = parser.parse_args()
    database = PluginDatabase(arguments.database)
    database.initialize()

    if arguments.command == "show":
        for alias, key in SETTING_ALIASES.items():
            value = database.get_setting(key, [])
            print(f"{alias}: {json.dumps(value, ensure_ascii=False)}")
        return
    if arguments.command == "set":
        database.set_setting(
            SETTING_ALIASES[arguments.setting], sorted(set(arguments.ids))
        )
        print(f"Updated {arguments.setting}.")
        return
    try:
        value = json.loads(arguments.value)
    except json.JSONDecodeError as error:
        parser.error(f"value must be valid JSON: {error.msg}")
    database.set_setting(arguments.key, value)
    print(f"Updated {arguments.key}.")


if __name__ == "__main__":
    main()
