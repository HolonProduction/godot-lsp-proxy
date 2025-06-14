# Godot LSP Proxy

> The script isn't well tested, use at your own risk.

Proxy for the Godot language server that performs transformation on file paths and uris
that are exchanged between server and client. Might be helpfull wenn running
the client in a WSL container but not Godot.

Configure desired paramters at the top of `proxy.py`:
- PROXY_HOST, PROXY_PORT: The proxy will listen here for connections from the client editor

- GODOT_HOST, GODOT_PORT: Godot LSP to forward messages to

- CLIENT_PATH, SERVER_PATH: Two path in the client/server file system that point to the same underlying location. The proxy will transform file locations based on these paths.

- TRANFORM_MARKUP: Transform paths in markup content. Might do unwanted replaces in edge cases, but makes links in documentation work.
