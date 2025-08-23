<H2>Interactive graph creation with sequence replay and decisions about which branch to play</H2>

sqlite3 is part of Python's standard library. But you need to install Flask to run it.

```
pip install Flask
```

Running. Go to the folder where app.py is and simply enter

```
python app.py
```

The server will start on localhost:5000

### Notes / behavior details

* The play logic now expects `#<id>` anywhere in the **description** field (case doesn't matter). Example: `#3` or `go to #12`.
* If multiple `#ID`s appear (e.g. `#2, #7` or `#2 then #7`) they will be tried in the order they appear; the first match that points to an actual outgoing neighbor will be used.
* If a `#ID` is present but doesn't correspond to any outgoing connection (maybe you added the branch later or used the wrong id), the code will fall back to the deterministic behavior (lowest connection id) so playback still continues.
* You can edit the description at any time to change which child will be chosen during PLAY — the next PLAY run will use the new `#ID` values.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Webapp/SVG%20graph%20with%20sequence%20playback/images/pic1.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Webapp/SVG%20graph%20with%20sequence%20playback/images/pic2.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Webapp/SVG%20graph%20with%20sequence%20playback/images/pic3.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Webapp/SVG%20graph%20with%20sequence%20playback/images/pic4.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Webapp/SVG%20graph%20with%20sequence%20playback/images/pic5.png?raw=true)

