"""Unit tests for the Task Manager CLI."""

import json
from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner

from app import (
    add,
    cli,
    complete,
    delete,
    edit,
    format_status,
    highlight_keyword,
    is_overdue,
    list_tasks,
    load_tasks,
    save_tasks,
    search_sort_key,
    stats,
    task_matches_keyword,
)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


class TestLoadTasks:
    def test_returns_empty_list_when_file_missing(self, isolated_tasks_file: Path) -> None:
        assert load_tasks() == []

    def test_loads_existing_tasks(self, isolated_tasks_file: Path) -> None:
        tasks = [{"id": 1, "name": "Test", "done": False}]
        isolated_tasks_file.write_text(json.dumps(tasks), encoding="utf-8")
        assert load_tasks() == tasks

    def test_returns_empty_list_on_corrupt_file(self, isolated_tasks_file: Path) -> None:
        isolated_tasks_file.write_text("not-json", encoding="utf-8")
        assert load_tasks() == []

    def test_returns_empty_list_when_file_contains_object(self, isolated_tasks_file: Path) -> None:
        isolated_tasks_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        assert load_tasks() == []


class TestSaveTasks:
    def test_saves_and_reloads(self, isolated_tasks_file: Path) -> None:
        tasks = [{"id": 1, "name": "Test", "done": False}]
        save_tasks(tasks)
        assert load_tasks() == tasks

    def test_saves_empty_list(self, isolated_tasks_file: Path) -> None:
        save_tasks([])
        assert load_tasks() == []


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------


class TestIsOverdue:
    def test_past_due_date_is_overdue(self) -> None:
        task = {"done": False, "due_date": "2000-01-01"}
        assert is_overdue(task) is True

    def test_future_due_date_is_not_overdue(self) -> None:
        task = {"done": False, "due_date": "2099-12-31"}
        assert is_overdue(task) is False

    def test_done_task_is_never_overdue(self) -> None:
        task = {"done": True, "due_date": "2000-01-01"}
        assert is_overdue(task) is False

    def test_no_due_date_is_not_overdue(self) -> None:
        task = {"done": False, "due_date": None}
        assert is_overdue(task) is False

    def test_invalid_due_date_is_not_overdue(self) -> None:
        task = {"done": False, "due_date": "not-a-date"}
        assert is_overdue(task) is False


class TestFormatStatus:
    def test_pending_task(self) -> None:
        task = {"done": False, "due_date": None}
        assert format_status(task).plain == "Pending"

    def test_done_task(self) -> None:
        task = {"done": True, "due_date": None}
        assert format_status(task).plain == "✓ Done"

    def test_overdue_task_takes_priority_over_pending(self) -> None:
        task = {"done": False, "due_date": "2000-01-01"}
        result = format_status(task)
        assert result.plain == "Overdue"
        assert result.style == "bold red"


# ---------------------------------------------------------------------------
# CLI commands via CliRunner
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestAddCommand:
    def test_add_creates_task(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["add", "Buy milk"])
        assert result.exit_code == 0
        tasks = load_tasks()
        assert len(tasks) == 1
        assert tasks[0]["name"] == "Buy milk"
        assert tasks[0]["done"] is False

    def test_add_sets_defaults(self, runner: CliRunner) -> None:
        runner.invoke(cli, ["add", "Task"])
        task = load_tasks()[0]
        assert task["priority"] == "medium"
        assert task["tags"] == []
        assert task["due_date"] is None
        assert task["description"] == ""
        assert "created_at" in task

    def test_add_with_options(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            ["add", "Deploy", "--priority", "high", "--due", "2099-12-31", "--tag", "work", "--tag", "devops"],
        )
        assert result.exit_code == 0
        task = load_tasks()[0]
        assert task["priority"] == "high"
        assert task["due_date"] == "2099-12-31"
        assert task["tags"] == ["work", "devops"]

    def test_add_empty_name_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["add", "   "])
        assert result.exit_code != 0
        assert load_tasks() == []

    def test_add_name_too_long_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["add", "x" * 201])
        assert result.exit_code != 0

    def test_add_invalid_due_date_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["add", "Task", "--due", "31-12-2099"])
        assert result.exit_code != 0
        assert load_tasks() == []

    def test_add_assigns_sequential_ids(self, runner: CliRunner) -> None:
        runner.invoke(cli, ["add", "First"])
        runner.invoke(cli, ["add", "Second"])
        ids = [t["id"] for t in load_tasks()]
        assert ids == [1, 2]


class TestListCommand:
    def test_list_empty(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No tasks" in result.output

    def test_list_shows_tasks(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "Buy groceries" in result.output
        assert "Deploy to production" in result.output

    def test_list_filter_pending(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list", "--status", "pending"])
        assert result.exit_code == 0
        assert "Write unit tests" not in result.output  # it's done

    def test_list_filter_done(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list", "--status", "done"])
        assert result.exit_code == 0
        assert "Write unit tests" in result.output
        assert "Buy groceries" not in result.output

    def test_list_filter_priority(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list", "--priority", "high"])
        assert result.exit_code == 0
        assert "Deploy to production" in result.output
        assert "Buy groceries" not in result.output

    def test_list_filter_tag(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list", "--tag", "personal"])
        assert result.exit_code == 0
        assert "Buy groceries" in result.output
        assert "Deploy to production" not in result.output

    def test_list_filter_overdue(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list", "--overdue"])
        assert result.exit_code == 0
        assert "Deploy to production" in result.output  # due 2020-01-01
        assert "Buy groceries" not in result.output

    def test_list_no_match_message(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list", "--tag", "nonexistent"])
        assert result.exit_code == 0
        assert "No tasks match" in result.output


class TestCompleteCommand:
    def test_complete_marks_done(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["complete", "1"])
        assert result.exit_code == 0
        task = next(t for t in load_tasks() if t["id"] == 1)
        assert task["done"] is True

    def test_complete_already_done(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["complete", "3"])  # task 3 is already done
        assert result.exit_code == 0
        assert "already" in result.output

    def test_complete_nonexistent_id_fails(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["complete", "999"])
        assert result.exit_code != 0


class TestEditCommand:
    def test_edit_name(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "1", "--name", "New name"])
        assert result.exit_code == 0
        task = next(t for t in load_tasks() if t["id"] == 1)
        assert task["name"] == "New name"

    def test_edit_priority(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "1", "--priority", "high"])
        assert result.exit_code == 0
        task = next(t for t in load_tasks() if t["id"] == 1)
        assert task["priority"] == "high"

    def test_edit_due_date(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "1", "--due", "2099-06-30"])
        assert result.exit_code == 0
        task = next(t for t in load_tasks() if t["id"] == 1)
        assert task["due_date"] == "2099-06-30"

    def test_edit_clear_due_date(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "2", "--due", ""])
        assert result.exit_code == 0
        task = next(t for t in load_tasks() if t["id"] == 2)
        assert task["due_date"] is None

    def test_edit_tags(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "1", "--tag", "home", "--tag", "urgent"])
        assert result.exit_code == 0
        task = next(t for t in load_tasks() if t["id"] == 1)
        assert task["tags"] == ["home", "urgent"]

    def test_edit_no_changes_warns(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "1"])
        assert result.exit_code == 0
        assert "No changes" in result.output

    def test_edit_nonexistent_id_fails(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "999", "--name", "Nope"])
        assert result.exit_code != 0

    def test_edit_invalid_due_date_fails(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "1", "--due", "bad-date"])
        assert result.exit_code != 0


class TestDeleteCommand:
    def test_delete_removes_task(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["delete", "1"])
        assert result.exit_code == 0
        ids = [t["id"] for t in load_tasks()]
        assert 1 not in ids

    def test_delete_nonexistent_id_fails(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["delete", "999"])
        assert result.exit_code != 0


class TestStatsCommand:
    def test_stats_empty(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "0" in result.output

    def test_stats_counts(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "3" in result.output  # total


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


class TestTaskMatchesKeyword:
    def test_matches_name_case_insensitively(self) -> None:
        task = {"name": "Deploy to Production", "description": ""}
        assert task_matches_keyword(task, "deploy") is True

    def test_matches_description_only(self) -> None:
        task = {"name": "Chore", "description": "Run the release pipeline"}
        assert task_matches_keyword(task, "release") is True

    def test_no_match_returns_false(self) -> None:
        task = {"name": "Buy groceries", "description": ""}
        assert task_matches_keyword(task, "deploy") is False

    def test_missing_description_key_is_safe(self) -> None:
        task = {"name": "Buy groceries"}
        assert task_matches_keyword(task, "groceries") is True

    def test_none_description_is_safe(self) -> None:
        task = {"name": "Buy groceries", "description": None}
        assert task_matches_keyword(task, "groceries") is True


class TestSearchSortKey:
    def test_orders_by_priority_then_id(self) -> None:
        tasks = [
            {"id": 3, "priority": "low"},
            {"id": 1, "priority": "high"},
            {"id": 2, "priority": "medium"},
            {"id": 4, "priority": "high"},
        ]
        ordered = sorted(tasks, key=search_sort_key)
        assert [t["id"] for t in ordered] == [1, 4, 2, 3]


class TestHighlightKeyword:
    def test_highlights_single_match(self) -> None:
        result = highlight_keyword("Deploy to production", "deploy")
        assert result.plain == "Deploy to production"
        spans = [s for s in result.spans if s.style == "bold yellow"]
        assert len(spans) == 1
        assert (spans[0].start, spans[0].end) == (0, 6)

    def test_highlights_multiple_matches_case_insensitively(self) -> None:
        result = highlight_keyword("Deploy the deployment DEPLOY", "deploy")
        spans = [s for s in result.spans if s.style == "bold yellow"]
        assert len(spans) == 3

    def test_no_keyword_returns_plain_text_unchanged(self) -> None:
        result = highlight_keyword("Some text", "")
        assert result.plain == "Some text"
        assert not any(s.style == "bold yellow" for s in result.spans)

    def test_empty_text_returns_em_dash(self) -> None:
        result = highlight_keyword("", "deploy")
        assert result.plain == "—"


# ---------------------------------------------------------------------------
# CLI commands via CliRunner (continued)
# ---------------------------------------------------------------------------


class TestSearchCommand:
    def test_search_matches_name(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["search", "deploy"])
        assert result.exit_code == 0
        assert "Deploy to production" in result.output
        assert "Buy groceries" not in result.output

    def test_search_matches_description(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["search", "release pipeline"])
        assert result.exit_code == 0
        assert "Deploy to production" in result.output

    def test_search_is_case_insensitive(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["search", "DEPLOY"])
        assert result.exit_code == 0
        assert "Deploy to production" in result.output

    def test_search_includes_pending_and_done_tasks(
        self, runner: CliRunner, sample_tasks: list[dict]
    ) -> None:
        result = runner.invoke(cli, ["search", "tests"])
        # "Write unit tests" (done) matches; ensure status column reflects it
        assert result.exit_code == 0
        assert "Done" in result.output

    def test_search_orders_by_priority_then_id(
        self, runner: CliRunner, isolated_tasks_file: Path
    ) -> None:
        tasks = [
            {
                "id": 1,
                "name": "alpha task low",
                "description": "",
                "priority": "low",
                "tags": [],
                "due_date": None,
                "done": False,
                "created_at": "2025-01-01T09:00:00",
            },
            {
                "id": 2,
                "name": "alpha task high",
                "description": "",
                "priority": "high",
                "tags": [],
                "due_date": None,
                "done": False,
                "created_at": "2025-01-01T09:00:00",
            },
            {
                "id": 3,
                "name": "alpha task medium",
                "description": "",
                "priority": "medium",
                "tags": [],
                "due_date": None,
                "done": False,
                "created_at": "2025-01-01T09:00:00",
            },
        ]
        isolated_tasks_file.write_text(json.dumps(tasks), encoding="utf-8")
        result = runner.invoke(cli, ["search", "alpha"])
        assert result.exit_code == 0
        pos_high = result.output.index("alpha task high")
        pos_medium = result.output.index("alpha task medium")
        pos_low = result.output.index("alpha task low")
        assert pos_high < pos_medium < pos_low

    def test_search_no_match_message(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["search", "nonexistent-keyword"])
        assert result.exit_code == 0
        assert "No tasks match your search." in result.output

    def test_search_empty_keyword_fails(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["search", ""])
        assert result.exit_code != 0

    def test_search_whitespace_keyword_fails(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["search", "   "])
        assert result.exit_code != 0

    def test_search_handles_missing_description(
        self, runner: CliRunner, isolated_tasks_file: Path
    ) -> None:
        tasks = [
            {
                "id": 1,
                "name": "task without description key",
                "priority": "medium",
                "tags": [],
                "due_date": None,
                "done": False,
                "created_at": "2025-01-01T09:00:00",
            }
        ]
        isolated_tasks_file.write_text(json.dumps(tasks), encoding="utf-8")
        result = runner.invoke(cli, ["search", "without"])
        assert result.exit_code == 0
        assert "task without description key" in result.output
