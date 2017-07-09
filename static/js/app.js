//Currently use static labels
var labels = ['Beauty', 'Education', 'Media', 'Grocery', 'Stationery',
'Investment', 'Dining', 'Transport', 'Entertainment', 'Clothing'];

var finhackApp = angular.module('finhackApp', ['ui.router']);

var transData;
socket.emit('getPersonalAnalysis', {name: 'Charles Davis', detail: true, label: 'None'}, function(data) {
  //console.log(data);
  transData = data;
});

var totalPrice;
socket.on('receipt', function(response){
  console.log(response);
  totalPrice = response;
});

var compData= {};
var ageGenderList = {'You':[], 'Other':[]};
var ageSalaryList = {'You':[], 'Other':[]};
$.getJSON("static/json/ageGender.json", function(json) {
  // compData = json;
  labels.forEach(function(label) {
    percentage = json[label]['percentage'];
    ageGenderList['You'].push(percentage);
    ageGenderList['Other'].push(1 - percentage);
  })
});

$.getJSON("static/json/ageSalary.json", function(json) {
  //console.log(json);
  // compData = json;
  labels.forEach(function(label) {

    percentage = json[label]['percentage']
    ageSalaryList['You'].push(percentage)
    ageSalaryList['Other'].push(1 - percentage)
  })
});
compData['age'] = ageGenderList;
compData['salary'] = ageSalaryList;

var prediction;
socket.emit('coupon', function(data) {
  prediction = data;
});

var scanTran;



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
  })
  .state('connect', {
    url: '/connect',
    views: {
      content: {
        templateUrl: 'static/partials/creditcard.html'
      }
    }
  })
  .state('login', {
    url: '/login',
    views: {
      content: {
        templateUrl: 'static/partials/login.html'
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
    //console.log(data);
    monthlyData = data;
    monthlyDiagram.addSeries({
        name: title,
        data: data
    });
  });
  $scope.localTransData = transData;
  $scope.labels = labels;
  function showTable (label) {
    $scope.drilledDown = true;
    $scope.chosenLabel = label;

    //Todo: get transaction data by category
    $scope.recordList = transData[label];
    //console.log("recordList", transData[label]);
  }

  monthlyDiagram = Highcharts.chart('monthlyDiagram', {
    chart: {
      plotBackgroundColor: null,
      plotBorderWidth: null,
      plotShadow: false,
      type: 'pie',
      events: {
        drilldown: function(e) {
          //console.log(e.point.name);
          socket.emit('getPersonalAnalysis', {name: 'Charles Davis', detail: false, label: e.point.name}, function(data) {
            //console.log("2", data[0][0]);
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
    colors: ['#E6EBE0', '#5CA4A9', '#FFE5D9', '#9D8189', '#FEFFA5',
    '#FFA0AC', '#ED6A5A', '#DAB6C4', '#7B886F', '#B4DC7F'],
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
    comparisonDiagram.setTitle({
      text: 'Comparison with people of your ' + $scope.chosenGroup
    }, false);
    var series = comparisonDiagram.series;
    for (var i = 0; i < series.length; i++) {
      if ($scope.chosenGroup == 'salary') {
        if (series[i].options.id == 'ageSalaryYou' || series[i].options.id == 'ageSalaryOthers') {
          series[i].setVisible(true);
        } else {
          series[i].setVisible(false);
        }
      } else {
        if (series[i].options.id == 'ageGenderYou' || series[i].options.id == 'ageGenderOthers') {
          series[i].setVisible(true);
        } else {
          series[i].setVisible(false);
        }
      }
    }
  }

  comparisonDiagram = Highcharts.chart('comparisonDiagram', {
    chart: {
        type: 'bar'
    },
    title: {
        text: 'Comparison with people of your ' + $scope.chosenGroup
    },
    xAxis: {
        categories: labels
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
        id: 'ageSalaryOthers',
        data: ageSalaryList['Other'],
        plotOptions: {
          enableMouseTracking: false,
          fillOpacity: 0.65
        },
        visible: false
    }, {
        name: 'You',
        id: 'ageSalaryYou',
        data: ageSalaryList['You'],
        dataLabels: {
          enabled: true,
          align: 'right',
          format: '<b>{point.percentage:.1f}%</b>'
        },
        visible: false
    },{
        name: 'Others',
        id: 'ageGenderOthers',
        data: ageGenderList['Other'],
        plotOptions: {
          enableMouseTracking: false,
          fillOpacity: 0.65,
        },
        visible: true
    }, {
        name: 'You',
        id: 'ageGenderYou',
        data: ageGenderList['You'],
        dataLabels: {
          enabled: true,
          align: 'right',
          format: '<b>{point.percentage:.1f}%</b>'
        },
        visible: true
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
