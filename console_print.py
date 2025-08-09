import json
import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def pretty_print(data):
    if RICH_AVAILABLE:  
        console = Console(force_terminal=True)
        console.print(
            Panel.fit(
                f"[bold cyan]{data['summary']}[/bold cyan]",
                title="Summary",
                style="bold green"
            )
        )

        # main summary
        # no wrap well cause- yes.
        table = Table(show_header=True, header_style="magenta")
        table.add_column("ID", style="cyan", justify="center", no_wrap=True)
        table.add_column("Type", style="yellow", no_wrap=True)
        table.add_column("Slides", style="green", no_wrap=True)
        table.add_column("Conflict Kind", style="red", no_wrap=False, max_width=20)
        table.add_column("Severity", style="bold red", no_wrap=True)
        table.add_column("Explanation", style="white", no_wrap=False, max_width=40)
        table.add_column("Suggested Fix", style="white", no_wrap=False, max_width=40)

        severity_colors = {"high": "bold red", "medium": "yellow", "low": "green"}

        for idx,item in enumerate(data["inconsistencies"]):
            slides_str = ", ".join(str(s) for s in item['slides'])
            sev_style = severity_colors.get(item["severity"].lower(), "white")
            table.add_row(
                f"[bold]{item['id']}[/bold]",
                item['type'],
                slides_str,
                item['conflict_kind'],
                f"[{sev_style}]{item['severity']}[/{sev_style}]",
                item['explanation'],
                item['suggested_fix']
            )
            if idx < len(data["inconsistencies"]) - 1:
                table.add_row("", "", "", "", "", "", "")

        console.print(table)
        #console.print("[white]" + "─" * console.width + "[/white]") 

        for item in data["inconsistencies"]:
            details_text = "\n".join(
                [f"[bold]Slide {stmt['slide']}:[/bold] {stmt['text']}" for stmt in item['statements']]
            )
            console.print(
                Panel.fit(details_text, title=f"Details for {item['id']}", style="blue")
            )
            console.print("")

    else:
        #incase no rich console available (generic 80 by 80 size)
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(data["summary"])
        print("\nINCONSISTENCIES:")
        for item in data["inconsistencies"]:
            slides_str = ", ".join(str(s) for s in item['slides'])
            print(f"\nID: {item['id']}")
            print(f"Type: {item['type']}")
            print(f"Slides: {slides_str}")
            print(f"Conflict Kind: {item['conflict_kind']}")
            print(f"Severity: {item['severity']}")
            print(f"Explanation: {item['explanation']}")
            print(f"Suggested Fix: {item['suggested_fix']}")
            print("Statements:")
            for stmt in item["statements"]:
                print(f"  Slide {stmt['slide']}: {stmt['text']}")
