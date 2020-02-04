import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.productsService';

@Component({
  selector: 'app-products',
  templateUrl: './products.component.html',
  styleUrls: ['./products.component.scss']
})
export class ProductsComponent implements OnInit { 

  products:any;
  Name: string;
  Description: string;
  Category: string;
  Withdrawn: string;
  Tags: string;

  view: number;
  status: string;
  sort: string;

  error:string = '';

  logged: boolean = false;


  constructor(private data: DataService) { }

  ngOnInit() {
    this.status = 'ACTIVE';
    this.sort = 'id|DESC';

    if (this.data.auth != "") {
      this.logged = true;
    }
    else {
      this.logged = false;
    }

    this.data.getProducts(this.status, this.sort)
      .subscribe(data => {
        this.products = data
        console.log(this.products);
        }
      );

      this.view = 20;



  }

  getClick() {
    this.data.getProducts(this.status, this.sort)
      .subscribe(data => {
        this.products = data
        console.log(this.products);
        console.log("AEK");
        }
      );
  }

  postClick() {
    this.data.postProducts(this.Name, this.Description, this.Category, this.Withdrawn, this.Tags)
      .subscribe( data => {
        console.log(data);
        if ((data as any).message != null) {
          this.error = (data as any).message;
        }
        else {
          this.error = '';
        }
        this.getClick();
        }
      );

  }

  deleteClick(id:string) {
    console.log('delete clicked');
    console.log(id);
    this.data.deleteProduct(id)
      .subscribe( data => {
        console.log(data);
        if ((data as any).message != null) {
          this.error = (data as any).message;
        }
        else {
          this.error = '';
        }
        this.getClick();
        }
      );
  }


}
