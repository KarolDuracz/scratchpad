<H2>Interactive graph creation with sequence replay and decisions about which branch to play</H2>

<i>The entire demo consists of only 4 files: app.py, graph.db and files in templates/index.html and admin.html. if you want to download it, download only these files.</i>

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

```
function chooseNextConnectionForNode(node) {
  const outs = getOutgoing(node.id);
  if (!outs || outs.length === 0) return null;
  if (outs.length === 1) return outs[0];

  // 1) Primary rule: look for one or more "#<id>" tokens in the node description.
  //    Example: "Follow #3" or "#1, #2" — it will try IDs in the order they appear.
  const desc = (node.description || '');
  const matches = Array.from(desc.matchAll(/\#\s*([0-9]+)/g)).map(m => Number(m[1]));

  if (matches.length > 0) {
    // Try each referenced ID in order and pick the outgoing connection that targets that ID.
    for (const wantedId of matches) {
      const found = outs.find(c => Number(c.target) === Number(wantedId));
      if (found) return found;
    }
    // If description referenced IDs but none match current outgoing targets,
    // we continue to fallback logic below (so you can add branches dynamically and update description).
  }

  // 2) (Legacy) If no #ID matched, fall back to deterministic choice:
  //    pick the outgoing connection with the smallest connection id (stable).
  outs.sort((a, b) => a.id - b.id);
  return outs[0];
}
```

I didn't describe this in the image, but you can certainly drag and drop nodes to change their position. Simply hover over a node and see the cursor change. Left-click and hold the mouse button, move it where you want, and then release the left button. The node is now in its new location, and it's saved in the database.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Webapp/SVG%20graph%20with%20sequence%20playback/images/pic1.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Webapp/SVG%20graph%20with%20sequence%20playback/images/pic2.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Webapp/SVG%20graph%20with%20sequence%20playback/images/pic3.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Webapp/SVG%20graph%20with%20sequence%20playback/images/pic4.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Webapp/SVG%20graph%20with%20sequence%20playback/images/pic5.png?raw=true)

