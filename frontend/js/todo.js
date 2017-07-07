
var finhackApp = angular.module('finhackApp', []);
finhackApp.controller('TodoCtrl', ['$scope',
  function($scope) {
    $scope.totalTodos = 4;
  }
]);
