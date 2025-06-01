from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("squirrel")

@mcp.tool()
async def get_name(characteristic: str) -> str:
    """Get a name for a squirrel based on a characteristic.

    Args:
       characteristic: A characteristic of the squirrel.
    """

    return f"{characteristic}Squirrel"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')



