from __future__ import annotations

import argparse

import uvicorn

from ragagent.config import get_settings
from ragagent.container import Container
from ragagent.worker import main as worker_main


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragagent")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("init", help="Initialize local data directories and database")
    serve = subcommands.add_parser("serve", help="Run the API and Web application")
    serve.add_argument("--reload", action="store_true")
    subcommands.add_parser("worker", help="Run the persistent ingestion/evaluation worker")
    args = parser.parse_args()
    settings = get_settings()
    if args.command == "init":
        Container(settings).initialize()
        print(f"Initialized Traceable Agentic RAG in {settings.data_dir.resolve()}")
    elif args.command == "serve":
        uvicorn.run(
            "ragagent.main:app",
            host=settings.host,
            port=settings.port,
            reload=args.reload,
        )
    else:
        worker_main()


if __name__ == "__main__":
    main()
