import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.productsService';

@Component({
  selector: 'app-nav',
  templateUrl: './nav.component.html',
  styleUrls: ['./nav.component.scss']
})

export class NavComponent implements OnInit {

  appTitle:string = 'ObservaGas';

  constructor(private data: DataService) { }

  ngOnInit() {
  }

}
