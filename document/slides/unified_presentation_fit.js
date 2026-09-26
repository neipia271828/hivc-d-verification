
  function fit() {
    const scale = Math.min(window.innerWidth / 1280, (window.innerHeight) / 720, 1);
    document.querySelector('.deck').style.transformOrigin = 'top left';
    document.querySelector('.deck').style.transform = 'scale(' + scale + ')';
    document.body.style.width = 1280 * scale + 'px';
  }
  window.addEventListener('resize', fit);
  window.addEventListener('beforeprint', () => {
    document.querySelector('.deck').style.transform = 'none';
    document.body.style.width = 'auto';
  });
  window.addEventListener('afterprint', fit);
  fit();
