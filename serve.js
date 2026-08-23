const http = require('http');
const fs = require('fs');
const path = require('path');
const dir = __dirname;
const types = { '.html': 'text/html', '.json': 'application/json', '.png': 'image/png', '.js': 'text/javascript' };
http.createServer((req, res) => {
  const file = path.join(dir, req.url === '/' ? 'index.html' : decodeURIComponent(req.url));
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
}).listen(4570, () => console.log('OK on http://127.0.0.1:4570'));
