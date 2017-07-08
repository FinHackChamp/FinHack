var finhackApp = angular.module('finhackApp', ['ui.router']);

finhackApp.config(function($stateProvider, $locationProvider) {
  $locationProvider.hashPrefix('!');
  $stateProvider.state('home', {
    url: '/',
    views: {
      content: {
        templateUrl: 'html/diagram.html',
        controller: 'DiagramCtrl'
      }
    }
  })
  .state('statistics', {
    url: '/statistics',
    views: {
      content: {
        templateUrl: 'html/statistics.html',
        controller: 'StatisticsCtrl'
      }
    }
  })
  .state('add', {
    url: '/add',
    onEnter: ['$stateParams', '$state', '$uibModal', function($stateParams, $state, $uibModal) {
      $uibModal.open({
          templateUrl: 'html/add.html',
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

  Highcharts.chart('monthlyDiagram', {
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

}]);
