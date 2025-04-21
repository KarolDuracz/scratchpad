<h2>A few minutes  with GPT 4o and I have it. Just WOW.</h2>

> This is to have a good point of reference of what this tool is. Today's model 4o is truly amazing at understanding and solving these types of problems. I'm giving a link to the entire session from chatgpt.
<br /><br />
Link to GPT session t- https://chatgpt.com/share/680618d8-42c0-8000-b762-81dc6536f7f4
<br /><br />

In a few minutes the model created quite a nice application that also works as a messenger. At the end of this session there is a change from a version that works only on localhost to a version that is visible through another computer in the LAN / WIFI network. And you can add TASK from another computer and after about 10 seconds it appears in the database. It is not only a planner but also a messenger on the network. SIMPLY AMAZING WHAT 4o CAN CREATE.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/21-04-2025%20-%20EurekaLabs%20practice/251%20-%2021-04-2025%20-%20WOW%20GPT%204o%20is%20amazing%20-%20fixed.png?raw=true)

1. <b>Just click on any week block at the top. This will show columns of tasks for the day. Sorted by time. By clicking on the task you can edit and change the state. There are 3 states PLANNED / DONE / NOTE. This also changes what the 10 upcoming tasks table will show. By clicking on "+" we add a task. If we change the status from PLANNED / NOTE to DONE then it disappears from the "top 10 table". Only planned or note are in this table. That's roughly how it works.</b>
2. In line app.py there is #cursor.execute('DROP TABLE IF EXISTS tasks') - This causes the database to be deleted every time the server is started. that's why it's commented out
3. app.py - last line 91 - app.run(debug=True, host='0.0.0.0') instead app.run(debug=True) so that the server is visible in LAN by other computers
4. index.html. Last lines are important in some way, because page refresh after 10 sec. If for example clinet 1 adds a new task then after about 10 seconds the rest of the clients (computers in the network) see this task on the page. Look at the link to chatgpt there at the end is given how to approach it using IO instead of polling.


```
loadTopTasks();
setInterval(loadTopTasks, 30000);

// for update insted of AJAX nad IO communication
setInterval(fetchTasks, 10000); // 10,000 ms = 10 seconds
```

<h2>I wonder what the models will be able to do in the next generations/years . Amazing tool, amazing job. :)</h2>
It's amazing that with my "weak english", AND ONLY ON THE BASIS OF TEXT, without seeing how it should look like, it generated a correct and efficient application. And this application is really useful in general after a few modifications. Planner + communicator is a tool used almost everywhere today, in every company. And at this level it's quite ok. And that's just based on the tokens that my poor English processed because these tokens are similar to others in the vocabulary... 
