from __future__ import annotations

import asyncio
import signal
from contextlib import suppress

from ragagent.config import get_settings
from ragagent.container import Container


async def run_worker() -> None:
    container = Container(get_settings())
    container.initialize()
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopped.set)
    while not stopped.is_set():
        job = container.repository.claim_next_job()
        if job:
            await container.job_processor.process(job)
            continue
        with suppress(TimeoutError):
            await asyncio.wait_for(stopped.wait(), timeout=0.5)


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
