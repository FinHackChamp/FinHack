
var finhackApp = angular.module('finhackApp', []);
finhackApp.controller('TodoCtrl', ['$scope',
  function($scope) {
    $scope.totalTodos = 4;
  }
]);

finhackApp.controller('DiagramCtrl', ['$scope',
  function($scope) {
    //Todo: Get data from Socket.io (backend)
    var jsonData = [{
      name: 'Education',
      y: 112.22
    }, {
      name: 'Media',
      y: 50.34
    }, {
      name: 'Investment',
      y: 20.96
    }];

    var title = moment().format("YYYY MMM");

    Highcharts.chart('container', {
      chart: {
          plotBackgroundColor: null,
          plotBorderWidth: null,
          plotShadow: false,
          type: 'pie'
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
        data: jsonData
      }]
    });
  }
]);
