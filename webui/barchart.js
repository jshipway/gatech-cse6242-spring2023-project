//uses svg, width, height, and margin global variables defined in globals.js
function drawBarChart (data) {

  if (svg!=null) {            
      d3.select("body").select("svg").remove();
  }

  document.getElementById("order_by_div").removeAttribute("hidden")

  svg = d3.select("body")
  .append("svg")
  .attr("id", "svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");
  

  // define tooltip
  d3.selectAll('.tooltip').remove() // delete any existing tooltips from previous searches

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
      .paddingInner(getBarPadding(data)[0])
      .paddingOuter(getBarPadding(data)[1])
      .align(0.1);

  var x = d3.scaleLinear();

  //blue scale
  var z = d3.scaleOrdinal().range(airport_colors);
      
  /*
  this is only used on the original quantile scale

  var riskScores = data.map((d) => d['Itinerary Risk']).sort(function(a,b) {return a-b})
  */

  dataInspect = data
  riskScale = d3.scaleThreshold()
                // domain is the itinerary risk as a percentage of the scheduled itinerary duration 
                .domain([0.05, 0.15, 0.25, 1.0])                                            
                .range(['#AFE1AF', '#fdcc8a', '#fc8d59', '#d7301f'])

  /*
  Original risk scale was quantile based. Requested to shift to a more consistent thresholding basis.
  Leaving the original code here as an option for later modelers.              
  riskScale = d3.scaleQuantile()
                          .domain(riskScores)                                              
                          //changed lowest risk to a light green, the prior light beige color was not showing well on US map
                          .range(['#AFE1AF', '#fdcc8a', '#fc8d59', '#d7301f'])
  */

    keys = ['Initial Flight', 'Connection Time', 'Final Flight']

    y.domain(data.map(function(d) { return d.ConnectCity; }));				
    x.rangeRound([0, width-legBuffer-circleSpacing - y.bandwidth()/2.5])	
      .domain([0, d3.max(data, function(d) { return d['Initial Flight']+d['Connection Time']+d['Final Flight']; })]).nice();	// y.domain...
    z.domain(keys);

    g.append("g")
      .attr("id", "bars")
      .selectAll("g")
      .data(d3.stack().keys(keys)(data))
      .enter().append("g")
        .attr("fill", function(d) { return z(d.key); })
      .selectAll("rect")
      .data(function(d) { return d; })
      .enter().append("rect")
        .attr("y", function(d) { return y(d.data.ConnectCity); })	    
        .attr("x", function(d) { return x(d[0]); })			    
        .attr("width", function(d) { return x(d[1]) - x(d[0]); })	
        .attr("height", y.bandwidth())
        .on('mouseover', function (d) {showRoute(d.data); return showTooltip(d)})
        .on('mouseout', function(d){normalRoute(d.data); return tooltip.style("visibility", "hidden")})	    
    
    var circles = g.append("g")
                  .attr("id", "risk_circles")

          var circRadius = y.bandwidth()/2.5
          circles.selectAll("circle")
                  .data(data)
                  .enter().append("circle")
                    .attr("class", "risk_circles")
                    .attr("cy", function(d) { return y(d.ConnectCity) + y.bandwidth()/2; })
                    .attr("cx", function(d) { return x(d['Initial Flight'] + d['Connection Time'] + d['Final Flight']) + d3.max([circleSpacing, circRadius + 5]);})
                    .attr("r", circRadius)
                    .attr('fill', function(d) { return riskScale(d['Itinerary Risk']) })
                    .attr('stroke', '#000000')
                    .on('mouseover', function(d){showRoute(d); return showTooltip(d)})
                    .on('mouseout', function (d) {normalRoute(d); return tooltip.style("visibility", "hidden") })
          
    /* This code prints the itinerary risk score to the bubbles
    This was requested to be removed by the project team on 4/10/2023. Keeping here in case others wish to resuscitate.

    var circleTxt = circles.append("g")
                            .attr("id", "risk_circle_text")

                  circleTxt.selectAll("text.risk_circle_text") 
                            .data(data)          
                            .enter().append("text")
                              .attr("class", "risk_circle_text")
                              .attr("y", function(d) { return y(d.ConnectCity) + y.bandwidth()/2; })
                              .attr("x", function(d) { return x(d['Initial Flight'] + d['Connection Time'] + d['Final Flight']) + d3.max([circleSpacing, circRadius + 5]);})
                              .attr('text-anchor', 'middle')
                              .attr('dominant-baseline', 'middle')
                              .attr('font-size', function(d) { return 24 - 1.25*document.getElementById("top_results").value})
                              .text(function(d) { return d['Itinerary Risk']})
    */

    // create y axis
    g.append("g")
        .attr("class", "axis")
        .attr("id", "y-axis")
        .attr("transform", "translate(0,0)") 						
        .call(d3.axisLeft(y));									
      
    d3.select("#y-axis").selectAll('text').each(insertLinebreaks);

    // create x axis
    g.append("g")
        .attr("class", "axis")
        .attr("id", "x-axis")
        .attr("transform", "translate(0,"+height+")")				
        .call(d3.axisBottom(x).ticks(null, "s"));					

      // add the x axis label    
      svg.append("text")
        .attr("id", "x_axis_label") 
        .attr("y", height + 50)												
        .attr("x", (width / 2) + 20) 						
        .attr("font-size", 10)        
        .attr("fill", "#000")        
        .attr("text-anchor", "middle")
        .text("Total Travel Duration (hrs)")
      
    var legend = g.append("g")
        .attr("font-family", "sans-serif")          
        .attr("font-size", 10)
        .attr("text-anchor", "end")
      .selectAll("g")
      .data(keys.slice()) 
      .enter().append("g")      
      .attr("transform", function(d, i) { return "translate(25," + (height/2 - 225 + i * 20) + ")"; });

    legend.append("rect")
        .attr("x", width - 19)
        .attr("width", 19)
        .attr("height", 19)
        .attr("fill", z);

    legend.append("text")
        .attr("x", width - 24)
        .attr("y", 9.5)
        .attr("dy", "0.32em")
        .text(function(d) { return d; });


    var bubbleLegend = g.append("g")

    var bubbleLegTitleHgt = 150
    bubbleLegend.append("text")
    .attr("x", width)
    .attr("y", bubbleLegTitleHgt)
    .attr("text-anchor", "middle")
    .attr('dominant-baseline', 'middle')
    .attr("font-size", 10) 
    .text('Itinerary Risk Level');

    var colors = ['#AFE1AF', '#fdcc8a', '#fc8d59', '#d7301f']
    var descriptions = ['Very Low', 'Low', 'Medium', 'High']
    var thresholdDesc = ['< 5%', '5-15%', '15-25%', '>25%']
    
    for (let i = 0; i < colors.length; i++){
      bubbleLegend.append("circle")
                  .attr("cx", width + 17)
                  .attr("cy", bubbleLegTitleHgt + 25 + 35*i)
                  .attr("r", 15)
                  .attr('stroke', '#000000')
                  .attr("fill", colors[i])
                  .on('mouseover', function(d){return showLegendInfo(thresholdDesc[i])})
                  .on('mouseout', function (d) {return tooltip.style("visibility", "hidden") })
        
      bubbleLegend.append("text")
                .attr("x", width - 3)
                .attr("y", bubbleLegTitleHgt + 25 + 35*i)
                .attr("text-anchor", "end")
                .attr('dominant-baseline', 'middle')
                .attr("font-size", 10) 
                .text(descriptions[i]);
    }
        
    function showTooltip(d){
      var top = d3.event.clientY + 5
      var left = d3.event.clientX + 5
      
      if ('data' in d){
        // this is the data structure for the rectangular bars
        var segmentLength = d[1] - d[0];

        tooltip.html(`Itinerary Segment Length:<br>${segmentLength.toFixed(1)} hours.`)
              .style("visibility", "visible")
              .style("left", left + "px")
              .style("top", top + "px")
              .style("opacity", 0.95)
              .style("color", "#fff")       
    }

      else{
        // if not rectangle the tooltip is hovering over one of the bubbles
        var chance = ( d['Chance of Missed Connection'] * 100 ).toFixed(1)
        if (chance < 0.1){
          // avoid committing to "zero" chance.
          // set to less than 0.1%.
          chance = '< 0.1'
        }
        var timeloss = d['Time Lost if Missed']
        
        tooltip.html(`Chance of Missed Connection: ${chance}%<br>Time Lost if Missed: ${timeloss} hours`)
              .style("visibility", "visible")
              .style("left", left + "px")
              .style("top", top + "px")
              .style("opacity", 0.95)
              .style("color", "#fff")
      }
    }

    function showLegendInfo(info){
      var top = d3.event.clientY - 5
      var left = d3.event.clientX - 5

      tooltip.html(`Expected delay is ${info} of the scheduled trip duration.`)
      .style("visibility", "visible")
      .style("left", left + "px")
      .style("top", top + "px")
      .style("opacity", 0.95)
      .style("color", "#fff")  

    }

    function showRoute(d) {
      var segid = "#seg" + d.FIRST_LEG_ORIG + d.FIRST_LEG_DEST;
      var segid2 = "#seg" + d.SECOND_LEG_ORIG + d.SECOND_LEG_DEST;
      riskColor = riskScale(d['Itinerary Risk'])
      d3.select("body").select("#svgmap").select(segid)
      .style("stroke", riskColor)  
      .attr("stroke-width", 6)

      d3.select("body").select("#svgmap").select(segid2)
      .style("stroke", riskColor)  
      .attr("stroke-width", 6)
    }


    function normalRoute(d) {
      var segid = "#seg" + d.FIRST_LEG_ORIG + d.FIRST_LEG_DEST;
      var segid2 = "#seg" + d.SECOND_LEG_ORIG + d.SECOND_LEG_DEST;
      d3.select("body").select("#svgmap").select(segid)
      .style("stroke", "black")  
      .attr("stroke-width", 2)

      d3.select("body").select("#svgmap").select(segid2)
      .style("stroke", "black")  
      .attr("stroke-width", 2)
    }


    function insertLinebreaks (d) {
      var el = d3.select(this);
      var lines = d.split('..');
      el.text('');
  
      for (var i = 0; i < lines.length; i++) {
          var tspan = el.append('tspan').text(lines[i]);
          if (i == 0){
            tspan.attr('x','-15').attr('y','-15')
          }
          else if (i > 0){
              tspan.attr('x','-15').attr('dy', '15');
          }
      }
    };

    function getBarPadding(obj){
      // given a json object (e.g., flights) return the inner and outer paddings
      // if five or more objects are requested to be returned, leave as default
      // if less than five, more care is provided to make the graph look more elegant.

      var numFlights = obj.length

      if (numFlights >= 5){
        var inner = 0.05
        var outer = 0
      }

      else{
        var inner = 0.2 + (4 - numFlights)*0.1
        var outer = 0.3 + (4 - numFlights)*0.2
      }

      return [inner, outer]
      
    }


}