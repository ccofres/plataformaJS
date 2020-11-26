const NavIndex = () => {
  const template = `
  <nav>
  <ul>
    <li><a href="./index.html">Home</a></li>
    <li><a href="./DLP/dlp.html">DLP</a></li>
    <li><a href="#">About</a></li>
    <li><a href="#">Contacto</a></li>
  </ul>
  </nav>
  `;
  return template;
};

const app = () => {
  document.getElementById("the-nav").innerHTML = NavIndex();
};
app();
