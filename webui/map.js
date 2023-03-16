//uses svgmap, width, height, and margin global variables defined in globals.js
function drawMap(flights) {

    d3.queue()
        .defer(d3.json, "./examples/us.json")
        .defer(d3.csv, "./examples/airports.csv", typeAirport)
        .defer(d3.csv, "./examples/flights.csv", typeFlight)
        .await(ready);
}
function ready(error, us, airports, flights) {
    if (error) throw error;

    if (svgmap!=null) {            
        d3.select("body").select("#svgmap").remove();
    }

    svgmap = d3.select("body").append("svg")
    .attr("id", "svgmap")
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

    flights = flights.slice(0, 30);

    var airportByIata = d3.map(airports, function(d) { return d.iata; });

    flights.forEach(function(flight) {
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

    svgmap.append("path")
        .datum({type: "MultiPoint", coordinates: airports})
        .attr("class", "airport-dots")
        .attr("d", path);

    var airport = svgmap.selectAll(".airport")
    .data(airports)
    .enter().append("g")
        .attr("class", "airport");

    airport.append("title")
        .text(function(d) { return d.iata + "\n" + d.arcs.coordinates.length + " flights"; });

    airport.append("path")
        .attr("class", "airport-arc")
        .attr("d", function(d) { return path(d.arcs); });

    airport.append("path")
        .data(voronoi.polygons(airports.map(projection)))
        .attr("class", "airport-cell")
        .attr("d", function(d) { return d ? "M" + d.join("L") + "Z" : null; });
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

