# gpx_formatter.py

TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="YourAppName - https://yourappurl.com"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xmlns="http://www.topografix.com/GPX/1/1"
     xsi:schemaLocation="http://www.topografix.com/GPX/1/1
                         http://www.topografix.com/GPX/1/1/gpx.xsd">
    <metadata>
        <name>{name}</name>
        <author>
            <name>Your Name or App</name>
        </author>
        <time>{timestamp}</time>
    </metadata>
    <trk>
        <name>{name}</name>
        <trkseg>
            {trace_points}
        </trkseg>
    </trk>
</gpx>"""

TRACE_POINT = """<trkpt lat="{lat}" lon="{lon}">
    <ele>{elevation}</ele>
    <time>{timestamp}</time>
</trkpt>"""