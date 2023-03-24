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
        flight2.origin = trip.SECOND_LEG_ORIG;
        flight2.destination = trip.SECOND_LEG_DEST;
        flight_array.push(flight1);
        flight_array.push(flight2);
      }

    origin_val = origin;
    dest_val = dest;

    d3.queue()
        .defer(d3.json, "./examples/us.json")
        .defer(d3.csv, "./examples/airports.csv", typeAirport)
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

    var radius = d3.scaleSqrt()
    .domain([0, 100])
    .range([0, 14]);

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

    // svgmap.append("path")
    //     .datum({type: "MultiPoint", coordinates: airports})
    //     .attr("class", "airport-dots")
    //     .attr("d", path);

    //draw airport dots, one by one 
    var dotcolors = ["#89cff0", "#0197f6", "#051094"];
    svgmap.selectAll(".airport-dots")
    .data(airports)
    .enter().append("circle", ".airport-dots")
    .attr("r", function(d) { 
        if (d.iata === origin_val) 
            return 8; 
        else if (d.iata === dest_val) 
            return 8;  
        else 
            return 4; ;  
        })
    .attr("transform", function(d) {
      return "translate(" + projection([
        d.longitude,
        d.latitude
      ]) + ")";
    })
    .attr("fill", function(d) { 
        if (d.iata === origin_val) 
            return dotcolors[0]; 
        else if (d.iata === dest_val) 
            return dotcolors[2];  
        else 
            return dotcolors[1]; ;  
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
            

    airport.append("path")
        .attr("class", "airport-arc")
        .attr("d", function(d) { return path(d.arcs); });

    airport.append("path")
        .data(voronoi.polygons(airports.map(projection)))
        .attr("class", "airport-cell")
        .attr("d", function(d) { return d ? "M" + d.join("L") + "Z" : null; });

    //create legend    
    mapkeys = ['Origin', 'Connection', 'Destination']
    var z = d3.scaleOrdinal().range(["#89cff0", "#0197f6", "#051094"]);
    z.domain(keys);

    var legend = svgmap.append("g")
        .attr("font-family", "sans-serif")          
        .attr("font-size", 10)
        .attr("text-anchor", "end")
      .selectAll("g")
      .data(mapkeys.slice()) //.reverse())
      .enter().append("g")
      //.attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });
      .attr("transform", function(d, i) { return "translate(220," + (height/2 - 50 + i * 20) + ")"; });

    // legend.append("rect")
    //     .attr("x", width - 19)
    //     .attr("width", 19)
    //     .attr("height", 19)
    //     .attr("fill", z);

    legend.append("circle")
        .attr("x", width - 19)
        .attr("r", 8)
        .attr("transform", "translate(680,9)")
        .attr("fill", z);

    legend.append("text")
        .attr("x", width - 24)
        .attr("y", 9.5)
        //.attr("font-size", 10)
        .attr("dy", "0.32em")
        .text(function(d) { return d; });


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

