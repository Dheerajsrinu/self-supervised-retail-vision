from passlib.context import CryptContext
import hashlib

_pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

def hash_password(password: str) -> str:
    # Pre-hash to fixed length
    password_bytes = password.encode("utf-8")
    sha256_hash = hashlib.sha256(password_bytes).hexdigest()
    return _pwd_context.hash(sha256_hash)

def verify_password(password: str, hashed: str) -> bool:
    password_bytes = password.encode("utf-8")
    sha256_hash = hashlib.sha256(password_bytes).hexdigest()
    return _pwd_context.verify(sha256_hash, hashed)
