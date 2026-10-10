"""Native dispatch and real MCP HTTP transport against a local protocol fixture."""
import asyncio
import json
import threading
import tempfile
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

from tools import parallel_search
from tools.tool_manager import ToolManager


class Handler(BaseHTTPRequestHandler):
    records = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(405)
        self.end_headers()

    def do_POST(self):
        message = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.records.append((self.path, dict(self.headers), message))
        method = message['method']
        if 'id' not in message:
            self.send_response(202)
            self.end_headers()
            return
        if method == 'initialize':
            result = {'protocolVersion': '2025-03-26', 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'parallel-fixture', 'version': '1'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': name, 'inputSchema': {'type': 'object'}}
                                for name in ['web_search', 'web_fetch']]}
        else:
            args = message['params']['arguments']
            if message['params']['name'] == 'web_search':
                data = {'results': [{'url': 'https://example.org/doc', 'title': 'Example',
                                     'publish_date': None, 'excerpts': ['Useful evidence.']}]}
            else:
                data = {'results': [{'url': url, 'excerpts': ['Page excerpt.']}
                                    for url in args['urls']], 'errors': []}
                if 'https://example.org/fail' in args['urls']:
                    data['errors'] = [{'url': 'https://example.org/fail', 'error_type': 'fetch_error'}]
            result = {'content': [{'type': 'text', 'text': json.dumps(data)}],
                      'structuredContent': data, 'isError': False}
            if args.get('objective') == 'tool error':
                result = {'content': [{'type': 'text', 'text': 'Rate limit reached'}], 'isError': True}
        payload = json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': result}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ParallelTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    async def asyncSetUp(self):
        Handler.records.clear()
        self.url_patch = patch.object(parallel_search, 'MCP_URL',
                                     f'http://127.0.0.1:{self.server.server_port}/mcp')
        self.url_patch.start()
        self.addCleanup(self.url_patch.stop)
        self.manager = await ToolManager.create(SimpleNamespace(
            dataset_name='gaia', enable_tool_search=False, research_web_provider='parallel',
        ))
        self.seq = {}

    async def call(self, name, arguments, manager=None, seq=None):
        return await (manager or self.manager).call_tool(
            {'function': {'name': name, 'arguments': arguments}}, self.seq if seq is None else seq)

    async def test_native_search_fetch_headers_contract_and_cache(self):
        self.manager.search_cache['query'] = [{'url': 'serper-result'}]
        self.manager.url_cache['https://example.org/doc'] = 'Jina content'
        results = await self.call('web_search', {'query': 'query'})
        self.assertEqual(results, [{'id': 1, 'title': 'Example', 'url': 'https://example.org/doc',
                                   'site_name': 'example.org', 'date': '', 'snippet': 'Useful evidence.'}])
        pages = await self.call('browse_pages', {'urls': ['https://example.org/doc']})
        self.assertEqual(pages, {'https://example.org/doc': 'Page excerpt.'})
        count = len(Handler.records)
        self.assertEqual(await self.call('web_search', {'query': 'query'}), results)
        self.assertEqual(await self.call('browse_pages', {'urls': ['https://example.org/doc']}), pages)
        self.assertEqual(len(Handler.records), count)
        self.assertEqual(self.manager.url_cache['https://example.org/doc'], 'Jina content')
        self.assertEqual(self.manager.search_cache['query'], [{'url': 'serper-result'}])
        methods = {row[2]['method'] for row in Handler.records}
        self.assertTrue({'initialize', 'tools/list', 'tools/call'} <= methods)
        calls = []
        for path, headers, message in Handler.records:
            self.assertEqual(path, '/mcp')
            normalized = {key.lower(): value for key, value in headers.items()}
            self.assertEqual(normalized['user-agent'], parallel_search.USER_AGENT)
            self.assertNotIn('authorization', normalized)
            if message['method'] == 'tools/call':
                calls.append(message['params'])
                self.assertEqual(message['params']['arguments']['session_id'], self.seq['parallel_session_id'])
        self.assertEqual([call['name'] for call in calls], ['web_search', 'web_fetch'])
        self.assertEqual(calls[0]['arguments']['search_queries'], ['query'])

    async def test_fetch_batching_and_deduplication(self):
        urls = [f'https://example.org/{i}' for i in range(21)]
        pages = await self.call('browse_pages', {'urls': urls + [urls[0]]})
        self.assertEqual(set(pages), set(urls))
        batches = [m['params']['arguments']['urls'] for _, _, m in Handler.records
                   if m['method'] == 'tools/call']
        self.assertEqual([len(batch) for batch in batches], [20, 1])

    async def test_saved_caches_keep_provider_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'search_cache.json').write_text(json.dumps({'query': [{'url': 'saved-serper'}]}))
            Path(directory, 'url_cache.json').write_text(json.dumps({'https://example.org/doc': 'saved-jina'}))
            args = SimpleNamespace(dataset_name='hle', enable_tool_search=False,
                                   research_web_provider='parallel', search_cache_dir=directory, url_cache_dir=directory)
            manager = await ToolManager.create(args)
            await self.call('web_search', {'query': 'query'}, manager)
            await self.call('browse_pages', {'urls': ['https://example.org/doc']}, manager)
            manager.save_caches()
            restored = await ToolManager.create(args)
            count = len(Handler.records)
            self.assertEqual((await self.call('web_search', {'query': 'query'}, restored))[0]['title'], 'Example')
            self.assertEqual(await self.call('browse_pages', {'urls': ['https://example.org/doc']}, restored),
                             {'https://example.org/doc': 'Page excerpt.'})
            self.assertEqual(len(Handler.records), count)
            self.assertEqual(restored.search_cache['query'], [{'url': 'saved-serper'}])
            self.assertEqual(restored.url_cache['https://example.org/doc'], 'saved-jina')

    async def test_errors_are_reported_and_not_cached(self):
        results = await self.call('web_search', {'query': 'tool error'})
        self.assertIn('Rate limit reached', results['error'])
        self.assertNotIn('parallel:tool error', self.manager.search_cache)
        pages = await self.call('browse_pages', {'urls': ['https://example.org/doc', 'https://example.org/fail']})
        self.assertIn('fetch_error', pages['error'])
        self.assertFalse(self.manager.url_cache)

    async def test_bad_arguments_do_not_make_requests(self):
        for arguments in [{'query': ''}, {'query': []}]:
            self.assertIn('error', await self.call('web_search', arguments))
        for urls in [[], [None], ['file:///tmp/secret']]:
            self.assertIn('error', await self.call('browse_pages', {'urls': urls}))
        self.assertFalse(Handler.records)

    async def test_all_research_datasets_and_distinct_sequences(self):
        for dataset in ['gaia', 'hle', 'browsecomp']:
            manager = await ToolManager.create(SimpleNamespace(
                dataset_name=dataset, enable_tool_search=False, research_web_provider='parallel'))
            seq = {}
            self.assertIsInstance(await self.call('web_search', {'query': dataset}, manager, seq), list)
            self.assertNotEqual(seq['parallel_session_id'], self.seq.get('parallel_session_id'))
            self.seq = seq

    async def test_timeout_and_cancellation(self):
        async def slow(*args):
            await asyncio.sleep(10)
        with patch('mcp.ClientSession.call_tool', side_effect=slow), patch.object(parallel_search, 'REQUEST_TIMEOUT', 0.05):
            result = await self.call('web_search', {'query': 'timeout'})
            self.assertIn('error', result)
            self.assertFalse(self.manager.search_cache)
        with patch('tools.parallel_search.parallel_search', side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await self.call('web_search', {'query': 'cancel'})

    async def test_default_serper_and_explicit_exa_stay_independent(self):
        google = ModuleType('tools.google_search')
        google.google_serper_search_async = AsyncMock(return_value=[])
        google.google_serper_search = lambda **kwargs: [{'url': 'serper', 'snippet': 'default'}]
        google.fetch_page_content = lambda **kwargs: {'serper': ('page', 'full page')}
        google.fetch_page_content_async = AsyncMock()
        google.extract_snippet_with_context = lambda *args: (False, '')
        exa = ModuleType('tools.exa_search')
        exa.exa_search_async = AsyncMock(return_value=[{'url': 'exa', 'snippet': 'explicit'}])
        exa.exa_search = lambda **kwargs: []
        with patch.dict('sys.modules', {'tools.google_search': google, 'tools.exa_search': exa}):
            default = await ToolManager.create(SimpleNamespace(dataset_name='gaia', enable_tool_search=False, google_serper_api='fixture'))
            self.assertEqual(await self.call('web_search', {'query': 'query'}, default), [{'url': 'serper', 'snippet': 'default'}])
            self.assertEqual(await self.call('browse_pages', {'urls': ['serper']}, default), {'serper': 'page'})
            self.manager.args.exa_api_key = 'fixture'
            self.assertEqual(await self.call('exa_search', {'query': 'query'}), [{'url': 'exa', 'snippet': 'explicit'}])
        self.assertFalse(Handler.records)


if __name__ == '__main__':
    unittest.main()
