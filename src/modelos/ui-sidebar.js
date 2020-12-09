function openNav() {
  document.getElementById("mySidenav").style.width = "250px";
}
function closeNav() {
  document.getElementById("mySidenav").style.width = "0";
}
function DenseNet() {
  closeNav();
  let list = document.getElementById("mySidenav");
  // As long as <ul> has a child node, remove it
  while (list.hasChildNodes()) {
    list.removeChild(list.firstChild);
  }

  let d1 = document.getElementById("mySidenav");
  d1.insertAdjacentHTML(
    "afterbegin",
    '<a href="javascript:void(0)" class="closebtn removeElement" onclick="closeNav()">&times;</a><a class="removeElement" href="../model-1/model-1.html" >Model 1</a><a class="removeElement" href="../model-2/model-2.html">Model 2</a><a class="removeElement" href="../model-3/model-3.html">Clasificador Iris</a><a class="removeElement" href="../model-4/model-4.html">Multiclasificador BCWD</a><a href="javascript:void(0)" class="closebtn removeElement" onclick="run()">&laquo;</a>'
  );
  time = setTimeout(openNav, 500);
}

function CnnNet() {
  closeNav();
  let list = document.getElementById("mySidenav");
  // As long as <ul> has a child node, remove it
  while (list.hasChildNodes()) {
    list.removeChild(list.firstChild);
  }

  //closeNav();
  let d1 = document.getElementById("mySidenav");
  d1.insertAdjacentHTML(
    "afterbegin",
    '<a href="javascript:void(0)" class="closebtn removeElement" onclick="closeNav()">&times;</a><a class="removeElement" href="../model-5/model-5.html">Clasificador MNIST</a><a class="removeElement" href="../model-6/model-6.html">Clasificador Fashion MNIST</a><a class="removeElement" href="../model-7/model-7.html">Transfer Learning con MobileNet</a><a class="removeElement" href="../model-8/model-8.html">Style Transfer con Magenta.js</a><a href="javascript:void(0)" class="closebtn removeElement" onclick="run()">&laquo;</a>'
  );
  time = setTimeout(openNav, 500);
}

function GenNet() {
  closeNav();
  let list = document.getElementById("mySidenav");
  // As long as <ul> has a child node, remove it
  while (list.hasChildNodes()) {
    list.removeChild(list.firstChild);
  }

  let d1 = document.getElementById("mySidenav");
  d1.insertAdjacentHTML(
    "afterbegin",
    '<a href="javascript:void(0)" class="closebtn removeElement" onclick="closeNav()">&times;</a><a class="removeElement" href="../model-8/model-8.html">GAN</a><a href="javascript:void(0)" class="closebtn removeElement" onclick="run()">&laquo;</a>'
  );
  time = setTimeout(openNav, 500);
}

function run() {
  closeNav();
  let list = document.getElementById("mySidenav");
  // As long as <ul> has a child node, remove it
  while (list.hasChildNodes()) {
    list.removeChild(list.firstChild);
  }
  let bar = document.getElementById("mySidenav");
  bar.insertAdjacentHTML(
    "afterbegin",
    '<a href="javascript:void(0)" class="closebtn removeElement" onclick="closeNav()">&times;</a><a href="javascript:void(0)" class="removeElement" onclick="DenseNet()">Redes Neuronales</a><a href="javascript:void(0)" class="removeElement" onclick="CnnNet()">Redes Neuronales Convolucionales</a><a class="removeElement" href="javascript:void(0)" onclick="GenNet()">Modelos Generativos</a>'
  );
  time = setTimeout(openNav, 500);
}
function firstTime() {
  let bar = document.getElementById("mySidenav");
  bar.insertAdjacentHTML(
    "afterbegin",
    '<a href="javascript:void(0)" class="closebtn removeElement" onclick="closeNav()">&times;</a><a href="javascript:void(0)" class="removeElement" onclick="DenseNet()">Redes Neuronales</a><a href="javascript:void(0)" class="removeElement" onclick="CnnNet()">Redes Neuronales Convolucionales</a><a class="removeElement" href="javascript:void(0)" onclick="GenNet()">Modelos Generativos</a>'
  );
}
document.addEventListener("DOMContentLoaded", firstTime);
