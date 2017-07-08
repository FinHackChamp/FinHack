var finhackApp = angular.module('finhackApp', ['ui.router']);

var transData;
socket.emit('getPersonalAnalysis', {name: 'Charles Davis', detail: true, label: 'None'}, function(data) {
  console.log(data);
  transData = data;
});

var compData;
var totalPrice;


//Todo: Test this data
/*
socket.emit('getComparison', {name: 'Charles Davis', criteria: 'age', target: 'percentage'}, function(data) {
  console.log(data);

  compData['age'] = data;
  for (int i = 0; i < data.length; i++) {
    compData['otherage'][i] = 1 - data[i];
  }
  console.log(compData['otherage']);

});
*/
//Todo: Test this data
/*
socket.emit('getComparison', {name: 'Charles Davis', criteria: 'salary', target: 'percentage'}, function(data) {
  console.log(data);

  compData['salary'] = data;
  for (int i = 0; i < data.length; i++) {
    compData['othersalary'][i] = 1 - data[i];
  }
  console.log(compData['othersalary']);

});
  */

var prediction;
socket.emit('coupon', function(data) {
  prediction = data;
});

var scanTran;

//Currently use static labels
var labels = ['Beauty', 'Beverages', 'Education', 'Media', 'Grocery', 'Stationary',
'Investment', 'Dining', 'Transport', 'Entertainment', 'Clothing'];

finhackApp.config(function($stateProvider) {
  $stateProvider.state('home', {
    url: '/',
    views: {
      content: {
        templateUrl: '/static/partials/diagram.html',
        controller: 'DiagramCtrl'
      }
    }
  })
  .state('statistics', {
    url: '/statistics',
    views: {
      content: {
        templateUrl: '/static/partials/statistics.html',
        controller: 'StatisticsCtrl'
      }
    }
  })
  .state('coupon', {
    url: '/coupon',
    views: {
      content: {
        templateUrl: '/static/partials/coupon.html',
        controller: 'CouponCtrl'
      }
    }
  });
});





// finhackApp.controller('AddCtrl', ['$scope', function($scope){
//   // $scope.totalPrice = totalPrice
//   console.log('here')
//   socket.on('receipt', function(response){
//       console.log(response)
//       $scope.totalPrice = response['total']
      
//   })
//   var add = function(){
//     $scope.$apply()
//   }
// }]);


finhackApp.controller('DiagramCtrl', ['$scope', function($scope) {
  var monthlyData;
  var title = "Summary of the Past 30 Days"
  //var title = moment().format("YYYY MMM");

  //Todo: Change the name to user name!
  socket.emit('getPersonalAnalysis', {name: 'Charles Davis', detail: false, label: 'None'}, function(data) {
    console.log(data);
    monthlyData = data;
    monthlyDiagram.addSeries({
        name: title,
        data: data
    });
  });
  $scope.transData = transData;
  $scope.labels = labels;
  function showTable (label) {
    $scope.drilledDown = true;
    $scope.chosenLabel = label;

    //Todo: get transaction data by category
    $scope.recordList = transData[label];
    console.log("recordList", transData[label]);
  }

  monthlyDiagram = Highcharts.chart('monthlyDiagram', {
    chart: {
      plotBackgroundColor: null,
      plotBorderWidth: null,
      plotShadow: false,
      type: 'pie',
      events: {
        drilldown: function(e) {
          console.log(e.point.name);
          socket.emit('getPersonalAnalysis', {name: 'Charles Davis', detail: false, label: e.point.name}, function(data) {
            console.log("2", data[0][0]);
            monthlyDiagram.addSeriesAsDrilldown(e.point, data[0][0]);
          });
          $scope.chosenLabel = e.point.name;
          showTable(e.point.name);
          $scope.$apply();
        },
        drillup: function() {
          $scope.drilledDown = false;
          $scope.$apply();
        }
      }
    },
    credits: false,
    title: {
        text: title
    },
    tooltip: {
        pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
    },
    colors: ['#DAB6C4', '#7B886F', '#B4DC7F', '#FEFFA5', '#FFA0AC', '#ED6A5A', '#F4F1BB',
    '#9BC1BC', '#5CA4A9', '#FFA0AC', '#E6EBE0', '#5CA4A9', '#FFE5D9', '#9D8189'],
    plotOptions: {
        pie: {
            allowPointSelect: true,
            cursor: 'pointer',
            dataLabels: {
                enabled: false
            },
            showInLegend: true
        }
    },
    series: [],
    drilldown: {
      series: []
    }
  });
}]);

finhackApp.controller('CouponCtrl', ['$scope', function($scope) {

  $scope.prediction = prediction;

}]);

finhackApp.controller('StatisticsCtrl', ['$scope', function($scope) {
  $scope.chosenGroup = 'age';

  $scope.onChosenGroupChange = function() {
    /*
    var series = comparisonDiagram.series;
    for (int i = 0; i < series.length(); i++) {
      if (series[i].name == 'You') {
        series[i].data = compData[$scope.chosenGroup];
      } else {
        var otherName = 'other' + $scope.chosenGroup;
        series[i].data = compData[otherName];
      }
    }
    */
    comparisonDiagram.setTitle({
      text: 'Comparison with people of your ' + $scope.chosenGroup
    });
  }

  comparisonDiagram = Highcharts.chart('comparisonDiagram', {
    chart: {
        type: 'bar'
    },
    title: {
        text: 'Comparison with people of your ' + $scope.chosenGroup
    },
    xAxis: {
        categories: ['Education', 'Media', 'Investment', 'Grocery']
    },
    yAxis: {
        min: 0,
        title: {
            text: 'Percentage among group'
        }
    },
    legend: {
        enabled: false
    },
    colors: ['#b3d3cf', '#5CA4A9'],
    plotOptions: {
      series: {
        stacking: 'percent'
      }
    },
    series: [{
        name: 'Others',
        data: [0.2, 0.4, 0.8, 0.3],
        plotOptions: {
          enableMouseTracking: false,
          fillOpacity: 0.65
        }
    }, {
        name: 'You',
        data: [0.8, 0.6, 0.2, 0.7],
        dataLabels: {
          enabled: true,
          align: 'right',
          format: '<b>{point.percentage:.1f}%</b>'
        }
    }]
  });
}]);

finhackApp.controller('AddCtrl', ['$scope', function($scope) {
  $scope.submitForm = function() {
    transData[$scope.addLabel].push({
      amount: $scope.addAmount,
      name: $scope.addDescription,
      time: $scope.addTime
    });
    //console.log("Pushed", transData);
  };
  $scope.showScanPane = function() {
    $scope.useScan = true;
  };
}]);
