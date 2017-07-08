var finhackApp = angular.module('finhackApp', []);

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

  Highcharts.chart('container', {
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
