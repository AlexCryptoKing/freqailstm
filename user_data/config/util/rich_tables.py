import sys
from typing import Any, Dict, List, Optional, Sequence, Union

from pandas import DataFrame
from rich.console import Console
from rich.table import Column, Table
from rich.text import Text


TextOrString = Union[str, Text]


def print_rich_table(
    tabular_data: Sequence[Union[Dict[str, Any], Sequence[TextOrString]]],
    headers: Sequence[str],
    summary: Optional[str] = None,
    *,
    table_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    table = Table(
        *[c if isinstance(c, Column) else Column(c, justify="right") for c in headers],
        title=summary,
        **(table_kwargs or {}),
    )

    for row in tabular_data:
        if isinstance(row, dict):
            table.add_row(
                *[
                    row[header] if isinstance(row[header], Text) else str(row[header])
                    for header in headers
                ]
            )

        else:
            row_to_add: List[Union[str, Text]] = [r if isinstance(r, Text) else str(r) for r in row]
            table.add_row(*row_to_add)

    console = Console(
        width=200 if "pytest" in sys.modules else None,
    )
    console.print(table)


def _format_value(value: Any, *, floatfmt: str) -> str:
    if isinstance(value, float):
        return f"{value:{floatfmt}}"
    return str(value)


def print_df_rich_table(
    tabular_data: DataFrame,
    headers: Sequence[str],
    summary: Optional[str] = None,
    *,
    show_index=False,
    index_name: Optional[str] = None,
    table_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    table = Table(title=summary, **(table_kwargs or {}))

    if show_index:
        index_name = str(index_name) if index_name else tabular_data.index.name
        table.add_column(index_name)

    for header in headers:
        table.add_column(header, justify="right")

    for value_list in tabular_data.itertuples(index=show_index):
        row = [_format_value(x, floatfmt=".3f") for x in value_list]
        table.add_row(*row)

    console = Console(
        width=200 if "pytest" in sys.modules else None,
    )
    console.print(table)
