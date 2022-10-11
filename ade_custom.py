    
class COLOR_PARAM:
    
    CLASSES = (
        'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar', 
        'motorcycle', 'bicycle', 'twoWheeler', 'pedestrian', 'rider', 'freespace',
        'curb', 'sidewalk', 'crossWalk', 'safetyZone', 'speedBump', 'roadMark', 'whiteLane',
        'yellowLane', 'blueLane', 'redLane', 'stopLane', 'constructionGuide', 'trafficDrum',
        'rubberCone', 'trafficSign', 'trafficLight', 'warningTriangle', 'fence'
    )
    
    def get_class_lowercase(self):
        return tuple(cls.lower() for cls in self.CLASSES)

    COLORMAP = [
                [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],  [128, 0, 128], [0, 128, 128], [128, 128, 128], 
                [64, 0, 0], [0, 64, 0], [0, 0, 64], [64, 64, 0],  [64, 0, 64], [0, 64, 64], [64, 64, 64], 
                [192, 0, 0], [0, 192, 0], [0, 0, 192], [192, 192, 0],  [192, 0, 192], [0, 192, 192], [192, 192, 192], 
                [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], 
                [192, 128, 128], [128, 64, 0], [0, 192, 128], [128, 192, 0], [0, 64, 128]
                    ]
