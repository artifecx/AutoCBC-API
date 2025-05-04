import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filename="requests.log",
    filemode="a"
)


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        # Gather request info
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        url = str(request.url)
        origin = request.headers.get("origin", "unknown")
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        logging.info(
            f"Request | time={now} | ip={client_ip} | method={method} | url={url} | origin={origin}"
        )

        response = await call_next(request)
        return response
