"""Gateway CLI — `python -m gateway.cli <command>`.

Day 1 has exactly one command that matters: `seed`, which creates the first user and
API key so the gateway is usable without curling `/admin/api-keys` by hand.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


async def seed(label: str = "default", email: str = "admin@example.com") -> int:
    """Create (or reuse) a user and issue their first API key.

    Idempotent-ish: if an active key already exists for the user, we print it? No —
    we *can't* print it, because only the hash is stored. So on re-run we issue a
    new key and revoke nothing; the operator picks whichever one they saved.
    """
    from sqlalchemy import select

    from gateway.config import get_settings
    from gateway.core.auth import generate_api_key, hash_api_key, key_prefix_of
    from gateway.db.models import ApiKey, User
    from gateway.db.session import dispose_db, get_session_factory, init_db

    settings = get_settings()
    await init_db()
    factory = get_session_factory()

    try:
        async with factory() as session:
            result = await session.execute(select(User).where(User.email == email))
            user = result.scalar_one_or_none()
            if user is None:
                user = User(email=email, display_name=email.split("@")[0], plan="team")
                session.add(user)
                await session.flush()
                print(f"created user #{user.id} ({email})")
            else:
                print(f"reusing user #{user.id} ({email})")

            plaintext = generate_api_key()
            key = ApiKey(
                user_id=user.id,
                label=label,
                key_prefix=key_prefix_of(plaintext),
                key_hash=hash_api_key(plaintext, settings.api_key_pepper),
                rate_limit_tokens_min=settings.rate_limit_capacity,
            )
            session.add(key)
            await session.commit()

            print()
            print("API key created — shown once, store it now:")
            print(f"  {plaintext}")
            print()
            print("Try it:")
            print(
                '  curl localhost:8000/v1/chat/completions '
                '-H "Authorization: Bearer '
                + plaintext
                + '" -H "Content-Type: application/json" '
                '-d \'{"model":"auto","messages":[{"role":"user","content":"hello"}]}\''
            )
            return 0
    finally:
        await dispose_db()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gateway.cli", description="LLM gateway CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    seed_p = sub.add_parser("seed", help="create a user + first API key")
    seed_p.add_argument("--label", default="default")
    seed_p.add_argument("--email", default="admin@example.com")

    args = parser.parse_args(argv)

    if args.command == "seed":
        return asyncio.run(seed(label=args.label, email=args.email))
    return 1  # pragma: no cover - argparse enforces subcommands


if __name__ == "__main__":
    sys.exit(main())
