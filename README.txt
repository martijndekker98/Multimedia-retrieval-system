For using the user interface to query models:
(Also see section 5.4 "User interface" in the report)
1. Run tkinterr_2.py
2. Select the querying method to use (custom distance function (with selected weights), KNN or RNN)
	2b. Select a K/R value (value <= 5)
3. Press the 'Go' button
4. Select the model to query and press 'Go' button
5. View the results



NOTE: on windows the viewer might crash (after inspecting a model), this is due to Vedo but we do not know what causes it.
On mac we did never have this crash.
The crash occurs when inspecting a model and then wanting to close this viewer.



For plotting the dimensionality reduction graph (using t-sne):
1. Run dimReduction.py
2. After a few second a window will be opened in the internet browser
PLOTLY INSTRUCTIONS
Zooming: select the area for zooming (double left mouse click or right mouse click to return to original view: zoom out)
Label and information will be shown when mouse hovers over point
Excluding class: click class in the legend on right side (pressing again will show the class again)
Focusing on class: double left mouse click on class (do again to show all classes again)
