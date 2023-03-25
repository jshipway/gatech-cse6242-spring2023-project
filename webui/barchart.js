//uses svg, width, height, and margin global variables defined in globals.js
function drawBarChart (data) {

  if (svg!=null) {            
      d3.select("body").select("svg").remove();
  }

  svg = d3.select("body")
  .append("svg")
  .attr("id", "svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  // width = +svg.attr("width") - margin.left - margin.right,
  // height = +svg.attr("height") - margin.top - margin.bottom,
  g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  // define tooltip
  var tooltip = d3.select("body")
                    .append("div")
                    .attr("class", "tooltip")
                    .style("visibility", "hidden")
                    .style("position", "absolute")
                    .style("background-color", "#666464")
                    .style("border", "solid")
                    .style("border-width", "1px")
                    .style("border-radius", "5px")
                    .style("padding", "10px")


    
  var y = d3.scaleBand()			// x = d3.scaleBand()	
      .rangeRound([0, height])	// .rangeRound([0, width])
      .paddingInner(0.05)
      .align(0.1);

  var x = d3.scaleLinear()		// y = d3.scaleLinear()
      .rangeRound([0, width-legBuffer]);	// .rangeRound([height, 0]);

  //blue scale
  var z = d3.scaleOrdinal()
      .range(["#b3cde0", "#0197f6", "#011f4b"]);

  var riskScores = data.map((d) => d['Itinerary Risk']).sort(function(a,b) {return a-b})
  //console.log(riskScores)
  //console.log((d) => d['Itinerary Risk'])

  dataInspect = data
  riskScale = d3.scaleQuantile()
                          .domain(riskScores)
                          .range(['#fef0d9', '#fdcc8a', '#fc8d59', '#d7301f'])

  // d3.csv("data.csv", function(d, i, columns) {
  //   for (i = 1, t = 0; i < columns.length; ++i) t += d[columns[i]] = +d[columns[i]];
  //   d.total = t;
  //   return d;
  // }, function(error, data) {
  //   if (error) throw error;

    //var keys = data.columns.slice(1);
    keys = ['Initial Flight', 'Connection Layover', 'Final Flight']

    //data.sort(function(a, b) { return b.total - a.total; });
    //data.sort(function(a, b) { return a.total - b.total; });
    y.domain(data.map(function(d) { return d.ConnectCity; }));					
    x.domain([0, d3.max(data, function(d) { return d['Initial Flight']+d['Connection Layover']+d['Final Flight']; })]).nice();	// y.domain...
    z.domain(keys);

    st = d3.stack().keys(keys)(data)
    console.log(st)

    g.append("g")
      .attr("id", "bars")
      .selectAll("g")
      .data(d3.stack().keys(keys)(data))
      .enter().append("g")
        .attr("fill", function(d) { return z(d.key); })
      .selectAll("rect")
      .data(function(d) { return d; })
      .enter().append("rect")
        .attr("y", function(d) { return y(d.data.ConnectCity); })	    //.attr("x", function(d) { return x(d.data.State); })
        .attr("x", function(d) { return x(d[0]); })			    //.attr("y", function(d) { return y(d[1]); })	
        .attr("width", function(d) { return x(d[1]) - x(d[0]); })	//.attr("height", function(d) { return y(d[0]) - y(d[1]); })
        .attr("height", y.bandwidth())						    //.attr("width", x.bandwidth());
    
    var circles = g.append("g")
                  .attr("id", "risk_circles")

          circles.selectAll("circle")
                  .data(data)
                  .enter().append("circle")
                    .attr("class", "risk_circles")
                    .attr("cy", function(d) { return y(d.ConnectCity) + y.bandwidth()/2; })
                    .attr("cx", function(d) { return x(d['Initial Flight'] + d['Connection Layover'] + d['Final Flight']) + circleSpacing;})
                    .attr("r", y.bandwidth()/2.5)
                    .attr('fill', function(d) { return riskScale(d['Itinerary Risk']) })
                    .attr('stroke', '#000000')
                    .on('mouseover', (d) => showTooltip(d))
                    .on('mouseout', function () { return tooltip.style("visibility", "hidden") })
          
    var circleTxt = circles.append("g")
                            .attr("id", "risk_circle_text")

                  circleTxt.selectAll("text.risk_circle_text") 
                            .data(data)          
                            .enter().append("text")
                              .attr("class", "risk_circle_text")
                              .attr("y", function(d) { return y(d.ConnectCity) + y.bandwidth()/2; })
                              .attr("x", function(d) { return x(d['Initial Flight'] + d['Connection Layover'] + d['Final Flight']) + circleSpacing;})
                              .attr('text-anchor', 'middle')
                              .attr('dominant-baseline', 'middle')
                              .attr('font-size', function(d) { return 24 - 1.25*document.getElementById("top_results").value})
                              .on('mouseover', (d) => showTooltip(d))
                              .text(function(d) { return d['Itinerary Risk']})


    // create y axis
    g.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0,0)") 						//  .attr("transform", "translate(0," + height + ")")
        .call(d3.axisLeft(y));									//   .call(d3.axisBottom(x));

    // create x axis
    g.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0,"+height+")")				// New line
        .call(d3.axisBottom(x).ticks(null, "s"));					//  .call(d3.axisLeft(y).ticks(null, "s"))

      // add the x axis label    
      svg.append("text")
        .attr("id", "x_axis_label") 
        .attr("y", height + 50)												//     .attr("y", 2)
        .attr("x", (width / 2) + 20) 						//     .attr("y", y(y.ticks().pop()) + 0.5)
        .attr("font-size", 10)
        //.attr("dy", "0.32em")										//     .attr("dy", "0.32em")
        .attr("fill", "#000")
        //.attr("font-weight", "bold")
        .attr("text-anchor", "middle")
        .text("Total Travel Duration (hrs)")
      
    var legend = g.append("g")
        .attr("font-family", "sans-serif")          
        .attr("font-size", 10)
        .attr("text-anchor", "end")
      .selectAll("g")
      .data(keys.slice()) //.reverse())
      .enter().append("g")
      //.attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });
      .attr("transform", function(d, i) { return "translate(25," + (height/2 - 225 + i * 20) + ")"; });

    legend.append("rect")
        .attr("x", width - 19)
        .attr("width", 19)
        .attr("height", 19)
        .attr("fill", z);

    legend.append("text")
        .attr("x", width - 24)
        .attr("y", 9.5)
        //.attr("font-size", 10)
        .attr("dy", "0.32em")
        .text(function(d) { return d; });
  //});



    function showTooltip(d){
      var top = d3.event.clientY + 5
      var left = d3.event.clientX + 5 
                      
      var chance = ( d['Chance of Missed Connection'] * 100 ).toFixed(2)
      var timeloss = d['Time Lost if Missed']

      tooltip.html(`Chance of Missed Connection: ${chance}%<br>Time Lost if Missed: ${timeloss} hrs.`)
            .style("visibility", "visible")
            .style("left", left + "px")
            .style("top", top + "px")
            .style("opacity", 0.95)
            .style("color", "#fff")
    }


}