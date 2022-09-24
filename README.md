# website classifier

classify websites based on their content

## usage

 * to classify a website
```
main.py --link 'https://hoogle.haskell.org'
```

 * to start the server
```
main.py --server
```
and send a POST request via `curl`
```
curl -X 'POST' 'http://localhost:5000/inference' -H 'Content-Type: application/json' -d '{"website_url": "https://hoogle.haskell.org"}'
```

