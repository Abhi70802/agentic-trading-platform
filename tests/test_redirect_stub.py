import http.client
import threading
import unittest

from redirect_stub import make_server


class RedirectStubTests(unittest.TestCase):
    def test_callback_discards_query_tokens_and_sets_no_store_headers(self):
        server = make_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            connection = http.client.HTTPConnection(host, port, timeout=3)
            connection.request("GET", "/angelone/callback?auth_token=secret&feed_token=secret")
            response = connection.getresponse()
            body = response.read()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Cache-Control"), "no-store")
            self.assertEqual(response.getheader("Referrer-Policy"), "no-referrer")
            self.assertNotIn(b"secret", body)
            self.assertIn(b"does not capture or store", body)
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_callback_refuses_non_loopback_binding(self):
        with self.assertRaisesRegex(ValueError, "loopback only"):
            make_server(host="0.0.0.0", port=8011)


if __name__ == "__main__":
    unittest.main()