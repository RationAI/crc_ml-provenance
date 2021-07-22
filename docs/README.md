# Pipeline Documentation

Documentation is accessible as an html website.
Follow these steps to view the website inside a browser.

0. Folder *docs/build/html/* should contain html files.  
If they are not present or you wish to update them, generate the files inside *docs/* folder by using command `make html`  


1. Start an http server inside *docs/build/html/* folder. e.g.:  
`python3 -m http.server`

2. Forward the used port (default is 8000) to your local machine  
`ssh -N -f -L localhost:8000:localhost:8000 <username>@<server_name>`

3. Visit `http://localhost:8000/`
