var finhackApp = angular.module('finhackApp', ['ui.router']);

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
  //Todo: Get data
  socket.emit('getPersonalAnalysis', {name: 'Rachel Trujillo'}, function(data) {
    console.log(data);
  });
  var jsonData = [{
    name: 'Education',
    y: 112.22,
    drilldown: 'Education'
  }, {
    name: 'Media',
    y: 50.34,
    drilldown: 'Media'
  }, {
    name: 'Investment',
    y: 20.96,
    drilldown: 'Investment'
  }];

  var title = moment().format("YYYY MMM");

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
            showInLegend: true
        }
    },
    series: [{
      name: title,
      data: jsonData
    }],
    drilldown: {
      series: [{
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
        data: [0.2, 0.4, 0.8, 0.3]
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

  $scope.onChosenGroupChange = function() {
    comparisonDiagram.setTitle({
      text: 'Comparison with people of your ' + $scope.chosenGroup
    });
  }
}]);
