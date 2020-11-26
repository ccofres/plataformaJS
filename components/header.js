const Header = () => {
  const template = `
  <header>
  <div class="leftheader">
    <a href="/"
      ><img src="img/logo.png" alt="Zorgblag's Blog. Click for home. "
    /></a>
  </div>
  <div class="rightheader">
    <h1>Plataforma JS</h1>
    <p>Un proyecto en Tensorflow.js</p>
  </div>
  </header>
  `;
  return template;
};

const app = () => {
  document.getElementById("the-header").innerHTML = Header();
};
app();
