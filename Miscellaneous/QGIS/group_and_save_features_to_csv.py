import csv
import os
from qgis.gui import QgsMapToolEmitPoint
from qgis.core import QgsFeatureRequest, QgsGeometry, QgsPointXY

class ClickTool(QgsMapToolEmitPoint):
    def __init__(self, canvas):
        self.canvas = canvas
        QgsMapToolEmitPoint.__init__(self, self.canvas)
        self.features_list = []
        self.save_id = 0
        self.filename = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\LM2\features_to_merge.csv"

    def canvasReleaseEvent(self, event):
        # Get the point where the mouse was clicked, in map coordinates
        map_point = self.toMapCoordinates(event.pos())
        
        # Assuming the active layer is a vector layer
        active_layer = self.canvas.currentLayer()
        
        # Buffer the point to create a small rectangle around it
        tolerance = 0.0001
        search_rect = QgsGeometry.fromPointXY(QgsPointXY(map_point.x() - tolerance, map_point.y() - tolerance)).boundingBox()
        search_rect.combineExtentWith(QgsGeometry.fromPointXY(QgsPointXY(map_point.x() + tolerance, map_point.y() + tolerance)).boundingBox())

        # Set up a query to find features within the rectangle
        query = QgsFeatureRequest().setFilterRect(search_rect)

        # Check each feature to see if it intersects the rectangle
        for feature in active_layer.getFeatures(query):
            geom = feature.geometry()
            if geom.intersects(QgsGeometry.fromRect(search_rect)):
                self.features_list.append(feature)
                print(f"Added feature {feature.id()} to list")

    def print_features_list(self):
        print("List of selected features:")
        for feature in self.features_list:
            print(f"Feature ID: {feature.id()}, Attributes: {feature.attributes()}")

    def save_features_to_csv(self):
        mode = 'a' if os.path.exists(self.filename) else 'w'
        with open(self.filename, mode, newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if mode == 'w':
                writer.writerow(['Group', 'Feature ID', 'Attributes'])  # Write header if file doesn't exist
            for feature in self.features_list:
                writer.writerow([self.save_id, feature.id(), feature.attributes()])

        print(f"Features appended to {self.filename}.")
        self.features_list = []  # Flush the list
        self.save_id += 1  # Increment the save ID for the next save

# Set the tool to your map canvas
canvas = iface.mapCanvas()
tool = ClickTool(canvas)
canvas.setMapTool(tool)
