var flight_array;
var origin_val;
var dest_val;

//uses svgmap, width, height, and margin global variables defined in globals.js
function drawMap(trips, origin, dest) {

    flight_array = [];    

    for (const trip of trips) {
        var flight1 = {};
        var flight2 = {};
        flight1.origin = trip.FIRST_LEG_ORIG;
        flight1.destination = trip.FIRST_LEG_DEST;
        flight1.risk_score = trip['Itinerary Risk'];
        flight2.origin = trip.SECOND_LEG_ORIG;
        flight2.destination = trip.SECOND_LEG_DEST;
        flight2.risk_score = trip['Itinerary Risk']
        flight_array.push(flight1);
        flight_array.push(flight2);
      }

    origin_val = origin;
    dest_val = dest;

    d3.queue()
        .defer(d3.json, "./data/us.json")
        .defer(d3.csv, "./data/airports.csv", typeAirport)
        .await(ready);
}
function ready(error, us, airports) { 
    if (error) throw error;

    if (svgmap!=null) {                    
        d3.select("body").select("#svgmap").remove();        
    }

    svgmap = d3.select("body").append("svg")
    .attr("id", "svgmap")
    .attr("x", margin.left)
    .attr("width", width + margin.left + margin.right)    
    .attr("height", height + margin.top + margin.bottom + 100)

    var projection = d3.geoAlbers()
    .translate([width / 2 + 100, height / 2 + 50])
    .scale(1280);

    var path = d3.geoPath()
    .projection(projection)
    .pointRadius(3.0);

    var voronoi = d3.voronoi()
    .extent([[-1, -1], [width + 1, height + 1]]);

    var airportByIata = d3.map(airports, function(d) { return d.iata; });

    flight_array.forEach(function(flight) {
        var source = airportByIata.get(flight.origin),
            target = airportByIata.get(flight.destination);
            source.arcs.coordinates.push([source, target]);
            target.arcs.coordinates.push([target, source]);
    });

    //get unique segments
    var unique_segments = [];    
    for (flight of flight_array) {
        var segment = findObjectByAttributes(unique_segments, 'origin', flight.origin, 'destination', flight.destination);
        if (segment == null) {
            origin = airportByIata.get(flight.origin);
            destination = airportByIata.get(flight.destination);
            segment = {};
            segment.origin = flight.origin;
            segment.destination = flight.destination;
            segment.id = "seg" + segment.origin + segment.destination;
            segment.origin_latitude = origin.latitude;
            segment.origin_longitutde = origin.longitude;
            segment.destination_latitude = destination.latitude;
            segment.destination_longitude = destination.longitude;
            segment.count = 1;
            segment.risk_score = flight.risk_score;
            unique_segments.push(segment)
        } 
        // else {
        //     //keep a running average of segment's risk score based on all the flights running through that path
        //     segment.risk_score = ((segment.risk_score * segment.count) + flight.risk_score) / (segment.count + 1)
        //     segment.count += 1;
        // }
    }


    airports = airports
        .filter(function(d) { return d.arcs.coordinates.length; });

    svgmap.append("path")
        .datum(topojson.feature(us, us.objects.land))
        .attr("class", "land")
        .attr("d", path);

    svgmap.append("path")
        .datum(topojson.mesh(us, us.objects.states, function(a, b) { return a !== b; }))
        .attr("class", "state-borders")
        .attr("d", path);

    //draw airport dots, one by one, prior version used a MultiPoint
    
    svgmap.selectAll(".airport-dots")
    .data(airports)
    .enter().append("circle", ".airport-dots")
    .attr("r", function(d) { 
        if (d.iata === origin_val) 
            return 10; 
        else if (d.iata === dest_val) 
            return 10;  
        else 
            return 6; ;  
        })
    .attr("transform", function(d) {
      return "translate(" + projection([
        d.longitude,
        d.latitude
      ]) + ")";
    })
    .attr("fill", function(d) { 
        if (d.iata === origin_val) 
            return airport_colors[0]; 
        else if (d.iata === dest_val) 
            return airport_colors[2];  
        else 
            return airport_colors[1]; ;  
        })
    


    var airport = svgmap.selectAll(".airport")
    .data(airports)
    .enter().append("g")
        .attr("class", "airport");

    airport.append("title")
        .text(function(d) { 
            if (d.iata === origin_val) 
                return d.iata + "\n" + d.arcs.coordinates.length + " outgoing flights"; 
            else if (d.iata === dest_val) 
                return d.iata + "\n" + d.arcs.coordinates.length + " incoming flights"; 
            else 
                return d.iata + "\n" + d.arcs.coordinates.length/2 + " connecting flights";  
            })
            
    //add routes to map
    svgmap.selectAll("line")
    .data(unique_segments)
    .enter().append("line", ".airport-arc")
    .attr("id", d=>d.id)
    .attr("x1", d=>projection([d.origin_longitutde, d.origin_latitude])[0])
    .attr("y1", d=>projection([d.origin_longitutde, d.origin_latitude])[1])
    .attr("x2", d=>projection([d.destination_longitude, d.destination_latitude])[0])
    .attr("y2", d=>projection([d.destination_longitude, d.destination_latitude])[1])
    .attr("stroke-width", 2)    
    //.attr('stroke', function(d) { return riskScale(d.risk_score);}  )
    .attr('stroke', 'black'  )

    airport.append("path")
        .data(voronoi.polygons(airports.map(projection)))
        .attr("class", "airport-cell")
        .attr("d", function(d) { return d ? "M" + d.join("L") + "Z" : null; });

    //create legend    
    mapkeys = ['Origin', 'Connection', 'Destination'] //#005b96
    var z = d3.scaleOrdinal().range(airport_colors);
    z.domain(mapkeys);

    var legend = svgmap.append("g")
        .attr("font-family", "sans-serif")          
        .attr("font-size", 10)
        .attr("text-anchor", "end")
      .selectAll("g")
      .data(mapkeys.slice()) 
      .enter().append("g")
      .attr("transform", function(d, i) { return "translate(220," + (height/2 - 50 + i * 20) + ")"; });

    legend.append("circle")
        .attr("x", width - 19)
        .attr("r", 8)
        .attr("transform", "translate(680,9)")
        .attr("fill", z);

    legend.append("text")
        .attr("x", width - 24)
        .attr("y", 9.5)        
        .attr("dy", "0.32em")
        .text(function(d) { return d; });
}

function findObjectByAttributes(array, attribute1, value1, attribute2, value2) {
    for (let i = 0; i < array.length; i++) {
      if (array[i][attribute1] === value1 && array[i][attribute2] === value2) {
        return array[i];
      }
    }
    return null;
  }

function typeAirport(d) {
  d[0] = +d.longitude;
  d[1] = +d.latitude;
  d.arcs = {type: "MultiLineString", coordinates: []};
  return d;
}

function typeFlight(d) {
  d.count = +d.count;
  return d;
}

