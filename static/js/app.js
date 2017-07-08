var finhackApp = angular.module('finhackApp', ['ui.router']);

compData = [];

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
        templateUrl: '/static/partials/coupon.html'
      }
    }
  })
  .state('add', {
    url: '/add',
    onEnter: ['$stateParams', '$state', '$uibModal', function($stateParams, $state, $uibModal) {
      $uibModal.open({
          templateUrl: 'static/partials/add.html',
          //controller: 'BondDialogController',
          //controllerAs: 'vm',
          backdrop: 'static',
          size: 'lg'
      }).result.then(function() {
          $state.go('home', null, { reload: 'home' });
      }, function() {
          $state.go('home');
      });
    }]
  });

});

finhackApp.controller('DiagramCtrl', ['$scope', function($scope) {
  var monthlyData;
  var title = "Summary of the Past 30 Days"
  //var title = moment().format("YYYY MMM");

  //Todo: Change the name to user name!
  socket.emit('getPersonalAnalysis', {name: 'Charles Davis'}, function(data) {
    console.log(data);
    monthlyData = data;
    monthlyDiagram.addSeries({
        name: title,
        data: data
    });
  });

  function showTable (label) {
    $scope.drilledDown = true;
    $scope.chosenLabel = label;

    //Todo: get transaction data by category
    $scope.recordList = [{
      time: new Date(),
      company: "Bullshit Co.",
      label: label,
      amount: 201.22
    }, {
      time: new Date(),
      company: "Apple Inc.",
      label: label,
      amount: 340.66
    }];
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
    plotOptions: {
        pie: {
            allowPointSelect: true,
            cursor: 'pointer',
            dataLabels: {
                enabled: false
            },
            showInLegend: true,
            colors: ['#DAB6C4', '#7B886F', '#B4DC7F', '#FEFFA5', '#FFA0AC', '#ED6A5A', '#F4F1BB',
            '#9BC1BC', '#5CA4A9', '#FFA0AC', '#E6EBE0', '#5CA4A9', '#FFE5D9', '#9D8189']
        }
    },
    series: [],
    drilldown: {
      series: [{
        //Todo: get Data!
        name: 'Education Details',
        id: 'Education',
        data: [{
          name: 'Coursera Inc.',
          y: 111.22
        }, {
          name: 'Pearson plc',
          y: 47.68
        }, {
          name: '2U, Inc.',
          y: 280.35
        }]
      }]
    }
  });
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
