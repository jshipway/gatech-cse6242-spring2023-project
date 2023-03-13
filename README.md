## Georgia Tech CSE 6242 - Data and Visual Analytics - Spring 2023 Project

# To run web application:

Once you clone the environment, cd into the webui directory
Once in that directory run:

python webServer.py

This runs the simplehttp server, with an attempt to address the cross site scripting issue we still have to work through.   The webserver that serves up our d3 web application will be running on port 8000 by default

Then cd into the dataManagement directory:
Once in that directory, run:

python itineraryBuilderService.py

This brings up our python back end service running on port 3000.   The d3 web application commuicates with this backend python service using the REST protocol.

You will need to start a browser that disables cross site scripting (CORS) checks until we figure out the configuration to allow this automatically for this web page and this python back end.   To disable CORS checks in Chrome, run chrome with the following parameters:

--disable-web-security --disable-gpu --user-data-dir="c:\temp"

On windows, you can make a shortcut with the following target:

"C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-web-security --disable-gpu --user-data-dir="c:\temp"

Your machine may not have a c:\temp, just put a valid temp directory there.

Now, open the browser, and in the address bar, enter:

http://localhost:8000/index.html


