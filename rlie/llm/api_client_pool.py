"""Thread-safe OpenAI-compatible client pool."""
import threading
from typing import List

from openai import OpenAI, AsyncOpenAI


class APIClientPool:
    """Thread-local synchronous clients plus round-robin async clients."""

    def __init__(self, base_url: str, api_keys: List[str]):
        if not api_keys:
            raise ValueError("APIClientPool requires at least one api key")

        self._base_url = base_url
        self._api_keys = api_keys
        self._lock = threading.Lock()
        self._idx = 0

        # Thread-local storage: each thread has its own client.
        self._thread_local = threading.local()

        # Async clients keep the original logic (asyncio has no thread contention).
        self._async_clients = [
            AsyncOpenAI(base_url=base_url, api_key=key) for key in api_keys
        ]
        self._lock_async = threading.Lock()
        self._async_idx = 0

    def get_sync_client(self) -> OpenAI:
        """
        Return the thread-local synchronous client.

        The first call on a thread creates a client for that thread;
        later calls reuse it.
        """
        # Check whether this thread already has a client.
        if not hasattr(self._thread_local, 'client'):
            with self._lock:
                # Assign a key to this thread (round-robin).
                key_idx = self._idx % len(self._api_keys)
                self._idx += 1

            # Create a client owned by this thread.
            self._thread_local.client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_keys[key_idx]
            )
            self._thread_local.key_idx = key_idx

        return self._thread_local.client

    def get_sync_client_for_retry(self, attempt: int) -> OpenAI:
        """
        Return a synchronous client for a retry, rotating the API key by attempt.

        Args:
            attempt: Current retry count (starting at 1).

        Returns:
            An OpenAI client instance.

        Each retry tries a different API key so a failing key does not block the request.
        """
        key_idx = (attempt - 1) % len(self._api_keys)
        return OpenAI(
            base_url=self._base_url,
            api_key=self._api_keys[key_idx]
        )

    def get_async_client(self) -> AsyncOpenAI:
        """
        Return an async client (original round-robin behavior).

        Async clients run under asyncio, so there is no thread contention.
        """
        with self._lock_async:
            client = self._async_clients[self._async_idx % len(self._async_clients)]
            self._async_idx += 1
            return client

    def get_async_client_for_retry(self, attempt: int) -> AsyncOpenAI:
        """
        Return an async client for a retry, rotating the API key by attempt.

        Args:
            attempt: Current retry count (starting at 1).

        Returns:
            An AsyncOpenAI client instance.

        Each retry tries a different API key so a failing key does not block the request.
        """
        key_idx = (attempt - 1) % len(self._api_keys)
        return self._async_clients[key_idx]

    def get_stats(self):
        """Return stats for debugging."""
        return {
            'num_api_keys': len(self._api_keys),
            'num_async_clients': len(self._async_clients),
            'total_sync_allocations': self._idx,
            'total_async_allocations': self._async_idx,
        }


# Backward compatibility for callers that used the cache.
class APIClientPoolWithCache(APIClientPool):
    """
    Cached variant, kept compatible with the original version.

    The cache matters less in this fix, because each thread creates its own client.
    It is kept mainly for backward compatibility.
    """

    _cache = {}
    _global_lock = threading.Lock()

    def __init__(self, base_url: str, api_keys: List[str]):
        normalized_keys = tuple(sorted(api_keys))
        cache_key = (base_url, normalized_keys)

        with APIClientPoolWithCache._global_lock:
            if cache_key in APIClientPoolWithCache._cache:
                # Take the existing pool from the cache.
                pool = APIClientPoolWithCache._cache[cache_key]
                # Copy references.
                self._base_url = pool._base_url
                self._api_keys = pool._api_keys
                self._lock = pool._lock
                self._idx = pool._idx
                self._thread_local = pool._thread_local
                self._async_clients = pool._async_clients
                self._lock_async = pool._lock_async
                self._async_idx = pool._async_idx
                return

            # Call the parent initializer.
            super().__init__(base_url, api_keys)
            # Store this pool in the cache.
            APIClientPoolWithCache._cache[cache_key] = self


if __name__ == "__main__":
    # Simple test.
    print("Testing the fixed APIClientPool")
    print("=" * 80)

    pool = APIClientPool(
        base_url="https://aihubmix.com/v1",
        api_keys=["key1", "key2", "key3"]
    )

    # Simulate several threads fetching clients.
    import threading

    client_ids = []
    lock = threading.Lock()

    def get_client(i):
        client = pool.get_sync_client()
        with lock:
            client_ids.append((i, id(client), threading.current_thread().name))

    threads = [threading.Thread(target=get_client, args=(i,), name=f"Thread-{i}") for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\nCreated {len(set(cid for _, cid, _ in client_ids))} distinct client instances")
    print("\nClient assignment:")
    for i, cid, tname in client_ids:
        print(f"  {tname}: client {cid}")

    print("\n✓ Each thread should have its own client instance")
    print(f"✓ Stats: {pool.get_stats()}")
