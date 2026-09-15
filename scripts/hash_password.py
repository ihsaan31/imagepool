"""Generate a bcrypt hash suitable for IMAGEPOOL_PASSWORD_HASH."""

from getpass import getpass

import bcrypt


password = getpass("Password: ")
confirmation = getpass("Confirm password: ")
if not password:
    raise SystemExit("Password cannot be empty")
if password != confirmation:
    raise SystemExit("Passwords do not match")

print(bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode())

