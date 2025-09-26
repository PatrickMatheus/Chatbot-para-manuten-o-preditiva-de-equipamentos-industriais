from mcp import ClientSession
from mcp.client.sse import sse_client
from contextlib import AsyncExitStack

class mcpUtil:
    def __init__(self):
        self.session = None
        self.async_exit_stack = AsyncExitStack()

    async def initialize_with_sse(self, host):
        self.client = await self.async_exit_stack.enter_async_context(sse_client(host))
        read, write = self.client 
        self.session = await self.async_exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()           

    async def call_tool(self, tool_name, args):
        return await self.session.call_tool(tool_name, arguments=args)
    
    async def get_resource(self, uri):
        return await self.session.read_resource(uri)
    
    async def invoke_prompt(self, prompt_name, args):
        return await self.session.get_prompt(prompt_name, arguments=args)
    
    async def get_tools(self):
        return (await self.session.list_tools()).tools
    
    async def get_resources(self, uri):
        return (await self.session.list_resources()).resources
    
    async def get_prompts(self):
        return (await self.session.list_prompts()).prompts
    
    async def clearup(self):
        await self.async_exit_stack.aclose()