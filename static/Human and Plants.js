var swiper = new Swiper('.s', {
    spaceBetween: 30,
    effect: 'fade',
    loop: true,
    mousewheel: {
      invert: false,
    },
    // autoHeight: true,
    pagination: {
      el: '.s__pagination',
      clickable: true,
    }
  });