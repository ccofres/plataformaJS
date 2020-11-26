function run() {
  let d1 = document.getElementById("mySidenav");
  d1.insertAdjacentHTML(
    "afterbegin",
    '<a href="javascript:void(0)" class="closebtn" onclick="closeNav()">&times;</a><a href="./modelos/model-1/model-1.html">Model 1</a><a href="./modelos/model-2/model-2.html">Model 2</a><a href="./modelos/model-3/model-3.html">Clasificador Iris</a><a href="./modelos/model-4/model-4.html">Multiclasificador BCWD</a><a href="./modelos/model-5/model-5.html">Clasificador MNIST</a><a href="./modelos/model-6/model-6.html">Clasificador Fashion MNIST</a><a href="./modelos/model-7/model-7.html">Transfer Learning con MobileNet</a><a href="./modelos/model-8/model-8.html">Style Transfer con Magenta.js</a><a href="./modelos/template-model/template-model.html">Template</a>'
  );
}

function openNav() {
  document.getElementById("mySidenav").style.width = "250px";
}
function closeNav() {
  document.getElementById("mySidenav").style.width = "0";
}

document.addEventListener("DOMContentLoaded", run);
